"""Build the small MPS stream bridge against the installed PyTorch ABI."""
from functools import cache
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import threading

import torch

_lock = threading.Lock()


@cache
def submission():
    with _lock:
        name = 'metalblas._mps_submission_native'
        if name in sys.modules:
            return sys.modules[name]
        source = Path(__file__).parent / 'native/submission.mm'
        torch_root = Path(torch.__file__).resolve().parent
        include = torch_root / 'include'
        stream_header = (include / 'ATen/mps/MPSStream.h').read_bytes()
        kernel_header = (include / 'ATen/native/mps/MetalShaderLibrary.h').read_bytes()
        libraries = tuple((name, (torch_root / 'lib' / name).stat().st_size,
                           (torch_root / 'lib' / name).stat().st_mtime_ns)
                          for name in ('libtorch_cpu.dylib', 'libtorch_python.dylib', 'libc10.dylib'))
        identity = repr((sys.implementation.cache_tag, sys.executable, str(torch_root),
                         torch.__version__, torch.version.git_version,
                         platform.machine(), platform.mac_ver()[0], libraries)).encode()
        digest = hashlib.sha256(source.read_bytes() + stream_header + kernel_header + identity).hexdigest()[:24]
        directory = Path.home() / 'Library/Caches/metalblas/native' / digest
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ('_mps_submission_native' + sysconfig.get_config_var('EXT_SUFFIX'))
        with (directory / 'build.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.is_file():
                temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
                command = ['xcrun', 'clang++', '-std=c++20', '-fblocks', '-shared', '-O2',
                           '-Wno-deprecated-declarations', '-Wno-objc-property-no-attribute',
                           '-I' + sysconfig.get_path('include'), '-I' + str(include),
                           '-I' + str(include / 'torch/csrc/api/include'),
                           '-L' + str(torch_root / 'lib'), '-ltorch_cpu', '-ltorch_python', '-lc10',
                           '-Wl,-rpath,' + str(torch_root / 'lib'), '-undefined', 'dynamic_lookup',
                           '-framework', 'Foundation', '-framework', 'Metal',
                           '-framework', 'MetalPerformanceShaders',
                           '-framework', 'MetalPerformanceShadersGraph',
                           str(source), '-o', str(temporary)]
                if b'setCurrentMPSStream(' in stream_header:
                    command.insert(2, '-DMB_HAS_MPS_STREAM_SETTER=1')
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError('Failed to build the MPS submission bridge against installed '
                                           'PyTorch. An Xcode C++ toolchain is required.\n' + result.stderr)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        spec = importlib.util.spec_from_file_location(name, target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        return module
