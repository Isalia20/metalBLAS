// Independent MPSGraph reference: avoids PyTorch's shape/layout-specific Metal fallbacks.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdio.h>

int mb_graph_addmm(const void *a, const void *b, const void *bias, const char *spec,
                   void *output, char *error, size_t error_size) {
    @autoreleasepool {
        @try {
            static id<MTLDevice> device;
            static id<MTLCommandQueue> queue;
            static dispatch_once_t once;
            dispatch_once(&once, ^{ device = MTLCreateSystemDefaultDevice(); queue = [device newCommandQueue]; });
            NSDictionary *s = [NSJSONSerialization JSONObjectWithData:
                [[NSString stringWithUTF8String:spec] dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
            MPSDataType types[] = {MPSDataTypeFloat32, MPSDataTypeFloat16, MPSDataTypeBFloat16};
            unsigned type = [s[@"dtype"] unsignedIntValue];
            MPSGraph *graph = [MPSGraph new];
            NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
            NSMutableArray *inputs = [NSMutableArray array];
            const void *pointers[] = {a, b, bias};
            NSArray *shapes = s[@"shapes"];
            for (unsigned i = 0; i < 3; ++i) {
                NSUInteger size = type == 0 ? 4 : 2;
                for (NSNumber *n in shapes[i]) size *= n.unsignedIntegerValue;
                id<MTLBuffer> buffer = [device newBufferWithBytes:pointers[i] length:size options:MTLResourceStorageModeShared];
                MPSGraphTensor *placeholder = [graph placeholderWithShape:shapes[i] dataType:types[type] name:nil];
                feeds[placeholder] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shapes[i] dataType:types[type]];
                [inputs addObject:placeholder];
            }
            MPSGraphTensor *aa = inputs[0], *bb = inputs[1], *x = inputs[2];
            if ([s[@"ta"] boolValue]) aa = [graph transposeTensor:aa dimension:aa.shape.count-2 withDimension:aa.shape.count-1 name:nil];
            if ([s[@"tb"] boolValue]) bb = [graph transposeTensor:bb dimension:bb.shape.count-2 withDimension:bb.shape.count-1 name:nil];
            MPSGraphTensor *result = [graph matrixMultiplicationWithPrimaryTensor:aa secondaryTensor:bb name:nil];
            if ([s[@"reduce_batch"] boolValue])
                result = [graph reductionSumWithTensor:result axis:0 name:nil];
            double alpha = [s[@"alpha"] doubleValue], beta = [s[@"beta"] doubleValue];
            if (alpha != 1.) result = [graph multiplicationWithPrimaryTensor:result secondaryTensor:
                [graph constantWithScalar:alpha dataType:types[type]] name:nil];
            if (beta != 0.) {
                if (beta != 1.) x = [graph multiplicationWithPrimaryTensor:x secondaryTensor:
                    [graph constantWithScalar:beta dataType:types[type]] name:nil];
                result = [graph additionWithPrimaryTensor:result secondaryTensor:x name:nil];
            }
            // Match PyTorch's MPSStream GPU compilation profile. The default
            // level may place half-precision operations on the Neural Engine.
            MPSGraphCompilationDescriptor *compilation = [MPSGraphCompilationDescriptor new];
            compilation.optimizationLevel = MPSGraphOptimizationLevel0;
            MPSGraphExecutionDescriptor *execution = [MPSGraphExecutionDescriptor new];
            execution.compilationDescriptor = compilation;
            execution.waitUntilCompleted = YES;
            MPSGraphTensorDataDictionary *values = [graph runAsyncWithMTLCommandQueue:queue feeds:feeds
                targetTensors:@[result] targetOperations:nil executionDescriptor:execution];
            [[values[result] mpsndarray] readBytes:output strideBytes:nil];
            return 0;
        } @catch (NSException *exception) {
            snprintf(error, error_size, "%s", exception.reason.UTF8String);
            return 1;
        }
    }
}
