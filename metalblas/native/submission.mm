// Encode source-selected launch plans atomically on the PyTorch stream.
// Submission limits come from the disassembled MPSGraph policy.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSProfiler.h>
#include <functional>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <pybind11/stl.h>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using Kernel = at::native::mps::MetalKernelFunction;
typedef struct { uint64_t operations, values, third; } MBDescriptor;
@interface MPSCommandBuffer (MBNativeSubmission)
- (MBDescriptor)mpsCommandBufferDescriptor;
- (void)setMpsCommandBufferDescriptor:(MBDescriptor)descriptor;
@end

@interface MPSGraphExecutionDescriptor (MBSourceSubmission)
@property(readonly) BOOL enableCommitAndContinue;
@end
@interface MPSGraphCompilationDescriptor (MBSourceSubmission)
@property(readonly) uint64_t compilerOptions;
@property(readonly) BOOL enableMemrefBufferization;
@end

// Use the public encode entry as an adapter to the installed Torch stream.
// No MPS graph operations are created, compiled, or executed here.
using EncodeFunction = std::function<void(MPSCommandBuffer*, MPSGraphExecutionDescriptor*)>;
@interface MBSubmissionGraph : MPSGraph {
@public
  EncodeFunction* sourceEncoder;
}
@end
@implementation MBSubmissionGraph
- (void)encodeToCommandBuffer:(MPSCommandBuffer*)buffer
                       feeds:(MPSGraphTensorDataDictionary*)feeds
            targetOperations:(NSArray<MPSGraphOperation*>*)targets
           resultsDictionary:(MPSGraphTensorDataDictionary*)results
         executionDescriptor:(MPSGraphExecutionDescriptor*)descriptor {
  TORCH_CHECK(sourceEncoder, "No source encoding invocation");
  (*sourceEncoder)(buffer, descriptor);
}
@end

namespace metalblas_submission {

struct BufferReference {
  uint64_t index, offset;
  BufferReference(uint64_t index_, uint64_t offset_) : index(index_), offset(offset_) {}
};

struct Argument {
  at::Tensor tensor;
  std::optional<BufferReference> reference;
  std::vector<uint8_t> bytes;
  template <class T> void assign(const std::vector<T>& values) {
    bytes.resize(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  explicit Argument(py::handle value) {
    if (py::isinstance<BufferReference>(value)) {
      reference = py::cast<BufferReference>(value);
    } else if (THPVariable_Check(value.ptr())) {
      tensor = THPVariable_Unpack(value.ptr());
      TORCH_CHECK(tensor.is_mps(), "Expected MPS tensor argument");
    } else if (py::isinstance<py::float_>(value)) {
      assign(std::vector<float>{py::cast<float>(value)});
    } else if (py::isinstance<py::int_>(value)) {
      assign(std::vector<int64_t>{py::cast<int64_t>(value)});
    } else if (py::isinstance<py::tuple>(value) || py::isinstance<py::list>(value)) {
      auto sequence = py::reinterpret_borrow<py::sequence>(value);
      TORCH_CHECK(sequence.size(), "Empty launch argument");
      if (py::isinstance<py::int_>(sequence[0])) {
        assign(py::cast<std::vector<int64_t>>(value));
      } else if (py::isinstance<py::float_>(sequence[0])) {
        assign(py::cast<std::vector<float>>(value));
      } else {
        TORCH_CHECK(false, "Unsupported launch argument sequence");
      }
    } else {
      TORCH_CHECK(false, "Unsupported launch argument");
    }
  }
};

// MPSAutoBuffer and these private MPSCore wrappers use the same cache. The cache
// owns returned objects; Return enables backing-store reuse after the final
// encoded use. See docs/mps_dispatch_sources.json for the recovered functions.
extern "C" id<MTLBuffer> MPSGetTemporaryBufferFromCommandBufferCache(id<MTLCommandBuffer>, NSUInteger);
extern "C" void MPSReturnTemporaryBufferToCommandBufferCache(id<MTLCommandBuffer>, id<MTLBuffer>);

struct BufferBinding {
  id<MTLBuffer> buffer = nil;
  uint64_t offset = 0, length = 0;
  bool graph_result = false, shared = false;
};

// MemrefBufferizer::createMemrefHeap uses shared, tracked placement heaps,
// default CPU caching, and alignment returned by the Metal device. Ordinary
// public addbmm has one graph intermediate, so its heap needs only one view.
struct GraphBuffer {
  id<MTLHeap> heap = nil;
  id<MTLBuffer> buffer = nil;
  GraphBuffer(id<MTLDevice> device, uint64_t length) {
    auto layout = [device heapBufferSizeAndAlignWithLength:length options:0];
    TORCH_CHECK(layout.align && layout.size >= length, "Invalid graph buffer layout");
    // ViewMemrefOpHandler rounds the view's size to the returned alignment.
    auto size = layout.size ? ((layout.size - 1) / layout.align + 1) * layout.align : 0;
    MTLHeapDescriptor* descriptor = [MTLHeapDescriptor new];
    descriptor.size = size;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    descriptor.type = MTLHeapTypePlacement;
    heap = [device newHeapWithDescriptor:descriptor];
    [descriptor release];
    TORCH_CHECK(heap, "Graph heap allocation failed");
    buffer = [heap newBufferWithLength:size options:0 offset:0];
    if (!buffer) {
      [heap release]; heap = nil;
      TORCH_CHECK(false, "Graph buffer allocation failed");
    }
  }
  ~GraphBuffer() { [buffer release]; [heap release]; }
};
using GraphBuffers = std::map<std::tuple<uintptr_t, uint64_t, uint64_t>, std::shared_ptr<GraphBuffer>>;

struct Buffers {
  // Retain caller tensors until encoding finishes. Encoder temporaries are
  // command-buffer-cache objects, allocated only when first referenced.
  std::vector<at::Tensor> inputs;
  std::vector<BufferBinding> bindings;
  std::vector<uint64_t> remaining;
  MPSCommandBuffer* command_buffer = nil;
  GraphBuffers* graph_buffers = nullptr;
  uintptr_t command_queue = 0;
  bool memref = false;
  Buffers(const std::vector<at::Tensor>& inputs_, const std::vector<uint64_t>& sizes,
          const std::vector<uint64_t>& uses, const std::vector<uint64_t>& graph_results)
      : inputs(inputs_), remaining(uses) {
    for (const auto& tensor : inputs)
      bindings.push_back({(id<MTLBuffer>)tensor.storage().data(),
                         static_cast<uint64_t>(tensor.storage_offset() * tensor.element_size()),
                         tensor.storage().nbytes()});
    for (auto size : sizes) bindings.push_back({nil, 0, size});
    for (auto index : graph_results) bindings.at(index).graph_result = true;
  }
  BufferBinding& get(uint64_t index) {
    auto& binding = bindings.at(index);
    if (!binding.buffer) {
      TORCH_CHECK(command_buffer && remaining.at(index), "Invalid temporary allocation");
      if (binding.graph_result && memref) {
        auto key = std::make_tuple(command_queue, index - inputs.size(), binding.length);
        auto& storage = (*graph_buffers)[key];
        if (!storage) storage = std::make_shared<GraphBuffer>(command_buffer.device, binding.length);
        binding.buffer = storage->buffer;
        binding.shared = true;
      } else {
        binding.buffer = MPSGetTemporaryBufferFromCommandBufferCache(command_buffer, binding.length);
      }
      TORCH_CHECK(binding.buffer, "MPS temporary allocation failed");
    }
    return binding;
  }
  void used(uint64_t index) {
    if (index >= inputs.size() && --remaining.at(index) == 0) {
      auto& binding = bindings.at(index);
      if (!binding.shared) MPSReturnTemporaryBufferToCommandBufferCache(command_buffer, binding.buffer);
      binding.buffer = nil;
    }
  }
  void release_pending() {
    for (size_t i = inputs.size(); i < bindings.size(); ++i) {
      if (bindings[i].buffer) {
        if (!bindings[i].shared) MPSReturnTemporaryBufferToCommandBufferCache(command_buffer, bindings[i].buffer);
        bindings[i].buffer = nil;
      }
    }
  }
};

struct Launch {
  std::shared_ptr<Kernel> kernel;
  std::vector<Argument> arguments;
  std::vector<uint64_t> threads, group;
  Launch(std::shared_ptr<Kernel> fn, py::tuple args,
         std::vector<uint64_t> grid, std::vector<uint64_t> tg)
      : kernel(std::move(fn)), threads(std::move(grid)), group(std::move(tg)) {
    TORCH_CHECK(threads.size() > 0 && threads.size() < 4 && group.size() == threads.size(),
                "Invalid launch dimensions");
    uint64_t size = 1;
    for (auto n : group) { TORCH_CHECK(n, "Empty threadgroup"); size *= n; }
    TORCH_CHECK(size <= kernel->getMaxThreadsPerThreadgroup(), "Threadgroup exceeds kernel limit");
    for (auto value : args) arguments.emplace_back(value);
  }
  void encode(Buffers& buffers) const {
    kernel->startEncoding();
    for (unsigned i = 0; i < arguments.size(); ++i) {
      const auto& arg = arguments[i];
      if (arg.reference) {
        const auto& ref = *arg.reference;
        const auto& binding = buffers.get(ref.index);
        auto offset = binding.offset + ref.offset;
        TORCH_CHECK(offset < binding.length, "Launch offset exceeds storage");
        auto encoder = at::mps::getCurrentMPSStream()->commandEncoder();
        [encoder setBuffer:binding.buffer offset:offset atIndex:i];
      } else if (arg.tensor.defined()) kernel->setArg(i, arg.tensor);
      else kernel->setArg(i, arg.bytes.data(), arg.bytes.size());
    }
    kernel->dispatch(threads, group);
    for (const auto& argument : arguments)
      if (argument.reference) buffers.used(argument.reference->index);
  }
};

struct Limits { uint64_t operations, values, maximum_operations, maximum_values; };
using Group = std::pair<std::vector<std::shared_ptr<Launch>>, uint64_t>;
struct Executable : std::enable_shared_from_this<Executable> {
  std::atomic<uint64_t> count{0};
  Limits zero, nonzero;
  MBSubmissionGraph* graph;
  // The same queue executes uses in order; different queues need distinct
  // backing storage. Scheduled/completed callbacks retain this owner through
  // the GPU's final use, even if Python evicts the executable meanwhile.
  GraphBuffers graph_buffers;
  Executable(std::array<uint64_t, 6> limits)
      : zero{limits[0], limits[1], limits[4], limits[5]},
        nonzero{limits[2], limits[3], limits[4], limits[5]}, graph([[MBSubmissionGraph alloc] init]) {}
  ~Executable() { [graph release]; }

  py::object run(const std::vector<Group>& groups, bool observe, Buffers& buffers) {
    auto* stream = at::mps::getCurrentMPSStream();
    std::exception_ptr error;
    uint64_t entry_count = 0, local_count = 0, commits = 0, compiler_options = 0;
    bool enabled = false, nested = false;
    MBDescriptor initial{}, seeded{}, local{};
    std::vector<std::array<uint64_t, 2>> steps, writes;
    EncodeFunction encode = [&](MPSCommandBuffer* buffer, MPSGraphExecutionDescriptor* descriptor) {
      // This must be a local value: an ObjC block formed from a reference
      // captured by the enclosing C++ lambda would retain that reference.
      auto state = shared_from_this();
      // Retain the calling stream's TLS selection inside dispatch_sync too.
#ifdef MB_HAS_MPS_STREAM_SETTER
      auto* previous = at::mps::getCurrentMPSStream();
      at::mps::setCurrentMPSStream(stream);
      struct Restore { at::mps::MPSStream* stream; ~Restore() { at::mps::setCurrentMPSStream(stream); } } restore{previous};
#endif
      enabled = descriptor.enableCommitAndContinue;
      compiler_options = descriptor.compilationDescriptor.compilerOptions;
      nested = (compiler_options & 0x800) != 0;
      TORCH_CHECK(buffer == stream->commandBuffer(), "Mismatched stream command buffer");
      [buffer addScheduledHandler:^(id<MTLCommandBuffer>) { ++state->count; }];
      entry_count = local_count = count.load();
      auto limits = local_count ? nonzero : zero;
      initial = local = [buffer mpsCommandBufferDescriptor];
      if (!local.operations) {
        local.operations = limits.operations >> 1;
        local.values = limits.values >> 1;
      }
      seeded = local;
      buffers.command_buffer = buffer;
      buffers.memref = (compiler_options & 0x4000) != 0 &&
          [descriptor.compilationDescriptor respondsToSelector:@selector(enableMemrefBufferization)];
      buffers.graph_buffers = &graph_buffers;
      buffers.command_queue = reinterpret_cast<uintptr_t>(stream->stream());
      try {
        for (const auto& [launches, produced] : groups) {
          for (const auto& launch : launches) launch->encode(buffers);
          if (enabled && produced) {
            auto before_ops = local.operations;
            ++local.operations;
            local.values += produced;
            limits = local_count ? nonzero : zero;
            bool commit = local.operations >= limits.maximum_operations || local.values >= limits.maximum_values ||
                          (local.operations >= limits.operations && local.values >= limits.values);
            if (commit) {
              stream->synchronize(at::mps::SyncType::COMMIT);
              local.operations = local.values = 0;
              ++commits;
            }
            if (local.operations < before_ops) ++local_count;
          }
          if (observe) steps.push_back({local.operations, local.values});
        }
        stream->synchronize(at::mps::SyncType::NONE);
        buffer = stream->commandBuffer();
        {
          [buffer setMpsCommandBufferDescriptor:local];
          if (observe) writes.push_back({local.operations, local.values});
          if (nested) {
            local = seeded;
            [buffer setMpsCommandBufferDescriptor:local];
            if (observe) writes.push_back({local.operations, local.values});
          }
        }
      } catch (...) {
        // Keep the scheduled/completed pair balanced if encoding throws.
        error = std::current_exception();
        buffers.release_pending();
        stream->synchronize(at::mps::SyncType::NONE);
      }
      [stream->commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer>) { --state->count; }];
    };
    // Keep the GIL, like torch.mps.MetalKernel.__call__: its shared kernel object
    // holds a mutable encoder, so concurrent Python launches must not race it.
    graph->sourceEncoder = &encode;
    struct ClearInvocation {
      MBSubmissionGraph* graph;
      ~ClearInvocation() { graph->sourceEncoder = nullptr; }
    } clear{graph};
    at::mps::getMPSProfiler().beginProfileKernel(graph, "metalblas_source_submission", true);
    @autoreleasepool {
      stream->executeMPSGraph(graph, nil, nil, at::mps::SyncType::COMMIT_ADAPTIVE);
    }
    if (error) std::rethrow_exception(error);
    if (!observe) return py::none();
    py::dict result;
    result["entry_count"] = entry_count;
    result["enabled"] = enabled;
    result["compiler_options"] = compiler_options;
    result["nested"] = nested;
    result["local_count"] = local_count;
    result["initial"] = std::array<uint64_t, 2>{initial.operations, initial.values};
    result["seeded"] = std::array<uint64_t, 2>{seeded.operations, seeded.values};
    result["steps"] = steps;
    result["writes"] = writes;
    result["commits"] = commits;
    result["memref"] = buffers.memref;
    return result;
  }
};

struct Program {
  std::vector<Group> groups;
  std::vector<uint64_t> sizes, uses;
  std::vector<uint64_t> graph_results;
  uint64_t input_count;
  Program(std::vector<Group> groups_, const std::vector<at::Tensor>& tensors, uint64_t count_,
          std::vector<uint64_t> graph_results_)
      : groups(std::move(groups_)), graph_results(std::move(graph_results_)), input_count(count_) {
    for (const auto& tensor : tensors) {
      TORCH_CHECK(tensor.is_meta(), "Expected metadata-only temporary");
      sizes.push_back(tensor.storage().nbytes());
    }
    uses.resize(input_count + sizes.size());
    for (auto index : graph_results)
      TORCH_CHECK(index >= input_count && index < uses.size(), "Invalid graph result index");
    for (const auto& [launches, produced] : groups)
      for (const auto& launch : launches)
        for (const auto& argument : launch->arguments)
          if (argument.reference) ++uses.at(argument.reference->index);
  }
  py::object run(const std::vector<at::Tensor>& inputs, const std::shared_ptr<Executable>& executable, bool observe) {
    TORCH_CHECK(inputs.size() == input_count && !inputs.empty(), "Invalid program inputs");
    for (const auto& input : inputs) TORCH_CHECK(input.is_mps(), "Expected MPS program input");
    Buffers buffers(inputs, sizes, uses, graph_results);
    return executable->run(groups, observe, buffers);
  }
};

} // namespace metalblas_submission

PYBIND11_MODULE(_mps_submission_native, m) {
  using namespace metalblas_submission;
  m.def("scalar_key", [](double value) { return std::to_string(value); });
  py::class_<BufferReference>(m, "BufferReference")
      .def(py::init<uint64_t, uint64_t>());
  py::class_<Program, std::shared_ptr<Program>>(m, "Program")
      .def(py::init<std::vector<Group>, const std::vector<at::Tensor>&, uint64_t, std::vector<uint64_t>>(),
           py::arg("groups"), py::arg("temporaries"), py::arg("input_count"),
           py::arg("graph_results") = std::vector<uint64_t>{})
      .def("run", &Program::run, py::arg("inputs"), py::arg("executable"), py::arg("observe") = false);
  py::class_<Launch, std::shared_ptr<Launch>>(m, "Launch")
      .def(py::init<std::shared_ptr<Kernel>, py::tuple, std::vector<uint64_t>, std::vector<uint64_t>>());
  py::class_<Executable, std::shared_ptr<Executable>>(m, "Executable")
      .def(py::init<std::array<uint64_t, 6>>())
      .def_property_readonly("count", [](Executable& e) { return e.count.load(); });
}
