// mbtrace.m - log Metal compute pipeline creation and dispatches (function names,
// function-constant values, bound bytes/buffers, grid sizes) as JSON lines.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <stdio.h>
#include <pthread.h>

static FILE *g_out;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static NSMutableDictionary *g_fcv;      // FCV ptr -> NSMutableArray of constant dicts
static NSMutableDictionary *g_fn;       // MTLFunction ptr -> info dict
static NSMutableDictionary *g_pso;      // PSO ptr -> info dict
static NSMutableDictionary *g_encPSO;   // encoder ptr -> PSO ptr
static NSMutableDictionary *g_encBytes; // encoder ptr -> {index: hex}
static NSMutableDictionary *g_encBufs;  // encoder ptr -> {index: desc}
static NSMutableDictionary *g_encTG;    // encoder ptr -> {index: len}
static NSMutableDictionary *g_libLabel; // library ptr -> label/url
static bool g_traceDAG;

static NSString *key(id o) { return [NSString stringWithFormat:@"%p", o]; }

static void emit(NSDictionary *d) {
    NSData *j = [NSJSONSerialization dataWithJSONObject:d options:0 error:nil];
    if (!j) return;
    pthread_mutex_lock(&g_lock);
    fwrite(j.bytes, 1, j.length, g_out); fputc('\n', g_out); fflush(g_out);
    pthread_mutex_unlock(&g_lock);
}

static NSString *hexOf(const void *p, NSUInteger n) {
    if (!p) return @"";
    NSUInteger m = n > 1024 ? 1024 : n;
    NSMutableString *s = [NSMutableString stringWithCapacity:m * 2];
    for (NSUInteger i = 0; i < m; i++) [s appendFormat:@"%02x", ((const unsigned char *)p)[i]];
    return s;
}

static NSString *typeName(MTLDataType t) {
    switch (t) {
        case MTLDataTypeBool: return @"bool"; case MTLDataTypeInt: return @"int"; case MTLDataTypeUInt: return @"uint";
        case MTLDataTypeShort: return @"short"; case MTLDataTypeUShort: return @"ushort"; case MTLDataTypeChar: return @"char";
        case MTLDataTypeUChar: return @"uchar"; case MTLDataTypeFloat: return @"float"; case MTLDataTypeHalf: return @"half";
        case MTLDataTypeLong: return @"long"; case MTLDataTypeULong: return @"ulong";
        default: return [NSString stringWithFormat:@"type%lu", (unsigned long)t];
    }
}
static NSUInteger typeSize(MTLDataType t) {
    switch (t) {
        case MTLDataTypeBool: case MTLDataTypeChar: case MTLDataTypeUChar: return 1;
        case MTLDataTypeShort: case MTLDataTypeUShort: case MTLDataTypeHalf: return 2;
        case MTLDataTypeInt: case MTLDataTypeUInt: case MTLDataTypeFloat: return 4;
        case MTLDataTypeLong: case MTLDataTypeULong: return 8;
        default: return 4;
    }
}
static id valueOf(const void *p, MTLDataType t) {
    switch (t) {
        case MTLDataTypeBool: return @(*(const bool *)p);
        case MTLDataTypeChar: return @(*(const int8_t *)p);
        case MTLDataTypeUChar: return @(*(const uint8_t *)p);
        case MTLDataTypeShort: return @(*(const int16_t *)p);
        case MTLDataTypeUShort: return @(*(const uint16_t *)p);
        case MTLDataTypeInt: return @(*(const int32_t *)p);
        case MTLDataTypeUInt: return @(*(const uint32_t *)p);
        case MTLDataTypeFloat: return @(*(const float *)p);
        case MTLDataTypeLong: return @(*(const int64_t *)p);
        case MTLDataTypeULong: return @(*(const uint64_t *)p);
        default: return hexOf(p, typeSize(t));
    }
}

// ---- generic hooking: replace every implementor of a selector with a block IMP ----
typedef id (*IMP0)(id, SEL);
static void hookAll(SEL sel, id (^maker)(IMP orig)) {
    unsigned n = 0;
    Class *list = objc_copyClassList(&n);
    for (unsigned i = 0; i < n; i++) {
        Class c = list[i];
        unsigned mc = 0;
        Method *ms = class_copyMethodList(c, &mc);
        for (unsigned k = 0; k < mc; k++) {
            if (method_getName(ms[k]) == sel) {
                IMP orig = method_getImplementation(ms[k]);
                IMP repl = imp_implementationWithBlock(maker(orig));
                method_setImplementation(ms[k], repl);
            }
        }
        free(ms);
    }
    free(list);
}

static NSMutableArray *constsFor(id fcv) {
    NSMutableArray *a = g_fcv[key(fcv)];
    if (!a) { a = [NSMutableArray array]; g_fcv[key(fcv)] = a; }
    return a;
}

static NSDictionary *fnInfo(id<MTLFunction> f) {
    NSDictionary *info = g_fn[key(f)];
    if (info) return info;
    return @{@"name": f.name ?: @"?", @"consts": @[]};
}

static void recordFunction(id<MTLFunction> f, NSString *name, id fcv, id lib) {
    if (!f) return;
    NSMutableDictionary *d = [NSMutableDictionary dictionary];
    d[@"name"] = name ?: (f.name ?: @"?");
    d[@"consts"] = fcv ? [constsFor(fcv) copy] : @[];
    d[@"lib"] = g_libLabel[key(lib)] ?: @"";
    g_fn[key(f)] = d;
    NSMutableDictionary *e = [d mutableCopy]; e[@"ev"] = @"function"; e[@"fn"] = key(f);
    emit(e);
}

static void recordPSO(id pso, id<MTLFunction> f, NSString *label) {
    if (!pso) return;
    NSMutableDictionary *d = [NSMutableDictionary dictionaryWithDictionary:fnInfo(f)];
    if (label) d[@"label"] = label;
    if ([pso respondsToSelector:@selector(maxTotalThreadsPerThreadgroup)])
        d[@"maxThreads"] = @([(id<MTLComputePipelineState>)pso maxTotalThreadsPerThreadgroup]);
    if ([pso respondsToSelector:@selector(threadExecutionWidth)])
        d[@"simdWidth"] = @([(id<MTLComputePipelineState>)pso threadExecutionWidth]);
    if ([pso respondsToSelector:@selector(staticThreadgroupMemoryLength)])
        d[@"staticTG"] = @([(id<MTLComputePipelineState>)pso staticThreadgroupMemoryLength]);
    g_pso[key(pso)] = d;
    NSMutableDictionary *e = [d mutableCopy]; e[@"ev"] = @"pso"; e[@"pso"] = key(pso);
    emit(e);
}

static NSMutableDictionary *sub(NSMutableDictionary *root, id enc) {
    NSMutableDictionary *d = root[key(enc)];
    if (!d) { d = [NSMutableDictionary dictionary]; root[key(enc)] = d; }
    return d;
}

static void emitDispatch(id enc, NSString *kind, MTLSize grid, MTLSize tg, NSDictionary *extra) {
    NSString *psoKey = g_encPSO[key(enc)];
    NSDictionary *pinfo = psoKey ? g_pso[psoKey] : nil;
    NSMutableDictionary *e = [NSMutableDictionary dictionary];
    e[@"ev"] = @"dispatch"; e[@"kind"] = kind;
    e[@"kernel"] = pinfo[@"name"] ?: @"?";
    e[@"consts"] = pinfo[@"consts"] ?: @[];
    e[@"label"] = pinfo[@"label"] ?: @"";
    e[@"lib"] = pinfo[@"lib"] ?: @"";
    e[@"grid"] = @[@(grid.width), @(grid.height), @(grid.depth)];
    e[@"tg"] = @[@(tg.width), @(tg.height), @(tg.depth)];
    e[@"bytes"] = [sub(g_encBytes, enc) copy];
    e[@"bufs"] = [sub(g_encBufs, enc) copy];
    e[@"tgmem"] = [sub(g_encTG, enc) copy];
    if (pinfo[@"maxThreads"]) e[@"maxThreads"] = pinfo[@"maxThreads"];
    if (pinfo[@"staticTG"]) e[@"staticTG"] = pinfo[@"staticTG"];
    if (extra) [e addEntriesFromDictionary:extra];
    emit(e);
}

__attribute__((constructor)) static void mbtrace_init(void) {
    const char *path = getenv("MBTRACE_OUT");
    g_out = fopen(path ? path : "/tmp/mbtrace.jsonl", "a");
    g_traceDAG = getenv("MBTRACE_DAG") != NULL;
    g_fcv = [NSMutableDictionary new]; g_fn = [NSMutableDictionary new]; g_pso = [NSMutableDictionary new];
    g_encPSO = [NSMutableDictionary new]; g_encBytes = [NSMutableDictionary new];
    g_encBufs = [NSMutableDictionary new]; g_encTG = [NSMutableDictionary new]; g_libLabel = [NSMutableDictionary new];

    // Force the concrete Metal classes to load before we enumerate implementors.
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc endEncoding];
    MTLFunctionConstantValues *fcv = [MTLFunctionConstantValues new];
    (void)fcv;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"kernel void _k(device float*o[[buffer(0)]],uint i[[thread_position_in_grid]]){o[i]=0;}" options:nil error:&err];
    (void)lib;

    // Function constant values
    hookAll(@selector(setConstantValue:type:atIndex:), ^id(IMP orig) {
        return ^(id self, const void *v, MTLDataType t, NSUInteger idx) {
            [constsFor(self) addObject:@{@"index": @(idx), @"type": typeName(t), @"value": valueOf(v, t)}];
            ((void (*)(id, SEL, const void *, MTLDataType, NSUInteger))orig)(self, @selector(setConstantValue:type:atIndex:), v, t, idx);
        };
    });
    hookAll(@selector(setConstantValue:type:withName:), ^id(IMP orig) {
        return ^(id self, const void *v, MTLDataType t, NSString *name) {
            [constsFor(self) addObject:@{@"name": name ?: @"?", @"type": typeName(t), @"value": valueOf(v, t)}];
            ((void (*)(id, SEL, const void *, MTLDataType, NSString *))orig)(self, @selector(setConstantValue:type:withName:), v, t, name);
        };
    });
    hookAll(@selector(setConstantValues:type:withRange:), ^id(IMP orig) {
        return ^(id self, const void *v, MTLDataType t, NSRange r) {
            NSUInteger sz = typeSize(t);
            for (NSUInteger i = 0; i < r.length; i++)
                [constsFor(self) addObject:@{@"index": @(r.location + i), @"type": typeName(t), @"value": valueOf((const char *)v + i * sz, t)}];
            ((void (*)(id, SEL, const void *, MTLDataType, NSRange))orig)(self, @selector(setConstantValues:type:withRange:), v, t, r);
        };
    });
    hookAll(@selector(reset), ^id(IMP orig) {
        return ^(id self) {
            if ([self isKindOfClass:[MTLFunctionConstantValues class]]) [g_fcv removeObjectForKey:key(self)];
            ((void (*)(id, SEL))orig)(self, @selector(reset));
        };
    });

    // Libraries (to tell shipped metallibs from runtime-compiled ones)
    hookAll(@selector(newLibraryWithURL:error:), ^id(IMP orig) {
        return ^id(id self, NSURL *url, NSError **e) {
            id r = ((id (*)(id, SEL, NSURL *, NSError **))orig)(self, @selector(newLibraryWithURL:error:), url, e);
            if (r) g_libLabel[key(r)] = [@"url:" stringByAppendingString:(g_traceDAG ? url.path : url.lastPathComponent) ?: @"?"];
            return r;
        };
    });
    hookAll(@selector(newLibraryWithFile:error:), ^id(IMP orig) {
        return ^id(id self, NSString *f, NSError **e) {
            id r = ((id (*)(id, SEL, NSString *, NSError **))orig)(self, @selector(newLibraryWithFile:error:), f, e);
            if (r) g_libLabel[key(r)] = [@"file:" stringByAppendingString:(g_traceDAG ? f : f.lastPathComponent) ?: @"?"];
            return r;
        };
    });
    hookAll(@selector(newLibraryWithData:error:), ^id(IMP orig) {
        return ^id(id self, dispatch_data_t d, NSError **e) {
            id r = ((id (*)(id, SEL, dispatch_data_t, NSError **))orig)(self, @selector(newLibraryWithData:error:), d, e);
            if (r) g_libLabel[key(r)] = [NSString stringWithFormat:@"data:%zu", dispatch_data_get_size(d)];
            return r;
        };
    });
    hookAll(@selector(newLibraryWithSource:options:error:), ^id(IMP orig) {
        return ^id(id self, NSString *src, id opts, NSError **e) {
            id r = ((id (*)(id, SEL, NSString *, id, NSError **))orig)(self, @selector(newLibraryWithSource:options:error:), src, opts, e);
            if (r) {
                g_libLabel[key(r)] = [NSString stringWithFormat:@"source:%lu", (unsigned long)src.length];
                emit(@{@"ev": @"source", @"lib": key(r), @"src": src ?: @""});
            }
            return r;
        };
    });

    // MPSCore uses the older DAG SPI, rather than MTLStitchedLibraryDescriptor,
    // for its linked prefix/postfix functions. Opt in to retaining the graph.
    if (g_traceDAG) hookAll(NSSelectorFromString(@"newLibraryWithDAG:functions:error:"), ^id(IMP orig) {
        return ^id(id self, id dag, NSArray *functions, NSError **e) {
            NSMutableArray *fs = [NSMutableArray array];
            for (id<MTLFunction> f in functions)
                [fs addObject:@{@"name": f.name ?: @"", @"info": fnInfo(f)}];
            emit(@{@"ev": @"dag", @"class": NSStringFromClass([dag class]),
                   @"description": [dag description], @"functions": fs});
            id r = ((id (*)(id, SEL, id, id, NSError **))orig)(
                self, NSSelectorFromString(@"newLibraryWithDAG:functions:error:"), dag, functions, e);
            if (r) g_libLabel[key(r)] = @"DAG";
            return r;
        };
    });

    // Functions
    hookAll(@selector(newFunctionWithName:constantValues:error:), ^id(IMP orig) {
        return ^id(id self, NSString *name, id fcv, NSError **e) {
            id r = ((id (*)(id, SEL, NSString *, id, NSError **))orig)(self, @selector(newFunctionWithName:constantValues:error:), name, fcv, e);
            recordFunction(r, name, fcv, self);
            return r;
        };
    });
    hookAll(@selector(newFunctionWithName:), ^id(IMP orig) {
        return ^id(id self, NSString *name) {
            id r = ((id (*)(id, SEL, NSString *))orig)(self, @selector(newFunctionWithName:), name);
            if ([self conformsToProtocol:@protocol(MTLLibrary)]) recordFunction(r, name, nil, self);
            return r;
        };
    });
    hookAll(@selector(newFunctionWithDescriptor:error:), ^id(IMP orig) {
        return ^id(id self, MTLFunctionDescriptor *desc, NSError **e) {
            id r = ((id (*)(id, SEL, MTLFunctionDescriptor *, NSError **))orig)(self, @selector(newFunctionWithDescriptor:error:), desc, e);
            recordFunction(r, desc.name, desc.constantValues, self);
            return r;
        };
    });

    // Pipelines
    hookAll(@selector(newComputePipelineStateWithFunction:error:), ^id(IMP orig) {
        return ^id(id self, id<MTLFunction> f, NSError **e) {
            id r = ((id (*)(id, SEL, id, NSError **))orig)(self, @selector(newComputePipelineStateWithFunction:error:), f, e);
            recordPSO(r, f, nil); return r;
        };
    });
    hookAll(@selector(newComputePipelineStateWithFunction:options:reflection:error:), ^id(IMP orig) {
        return ^id(id self, id<MTLFunction> f, MTLPipelineOption o, id *refl, NSError **e) {
            id r = ((id (*)(id, SEL, id, MTLPipelineOption, id *, NSError **))orig)(self, @selector(newComputePipelineStateWithFunction:options:reflection:error:), f, o, refl, e);
            recordPSO(r, f, nil); return r;
        };
    });
    hookAll(@selector(newComputePipelineStateWithDescriptor:options:reflection:error:), ^id(IMP orig) {
        return ^id(id self, MTLComputePipelineDescriptor *d, MTLPipelineOption o, id *refl, NSError **e) {
            id r = ((id (*)(id, SEL, id, MTLPipelineOption, id *, NSError **))orig)(self, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), d, o, refl, e);
            recordPSO(r, d.computeFunction, d.label); return r;
        };
    });
    hookAll(@selector(newComputePipelineStateWithFunction:completionHandler:), ^id(IMP orig) {
        return ^(id self, id<MTLFunction> f, void (^h)(id pso, NSError *err)) {
            ((void (*)(id, SEL, id, id))orig)(self, @selector(newComputePipelineStateWithFunction:completionHandler:), f,
                ^(id pso, NSError *err) { recordPSO(pso, f, nil); if (h) h(pso, err); });
        };
    });
    hookAll(@selector(newComputePipelineStateWithFunction:options:completionHandler:), ^id(IMP orig) {
        return ^(id self, id<MTLFunction> f, MTLPipelineOption o, void (^h)(id pso, id refl, NSError *err)) {
            ((void (*)(id, SEL, id, MTLPipelineOption, id))orig)(self, @selector(newComputePipelineStateWithFunction:options:completionHandler:), f, o,
                ^(id pso, id refl, NSError *err) { recordPSO(pso, f, nil); if (h) h(pso, refl, err); });
        };
    });
    hookAll(@selector(newComputePipelineStateWithDescriptor:options:completionHandler:), ^id(IMP orig) {
        return ^(id self, MTLComputePipelineDescriptor *d, MTLPipelineOption o, void (^h)(id pso, id refl, NSError *err)) {
            id<MTLFunction> f = d.computeFunction; NSString *label = d.label;
            ((void (*)(id, SEL, id, MTLPipelineOption, id))orig)(self, @selector(newComputePipelineStateWithDescriptor:options:completionHandler:), d, o,
                ^(id pso, id refl, NSError *err) { recordPSO(pso, f, label); if (h) h(pso, refl, err); });
        };
    });

    // Compute encoder state
    hookAll(@selector(setComputePipelineState:), ^id(IMP orig) {
        return ^(id self, id pso) {
            g_encPSO[key(self)] = key(pso);
            [g_encBytes removeObjectForKey:key(self)]; [g_encBufs removeObjectForKey:key(self)]; [g_encTG removeObjectForKey:key(self)];
            ((void (*)(id, SEL, id))orig)(self, @selector(setComputePipelineState:), pso);
        };
    });
    hookAll(@selector(setBytes:length:atIndex:), ^id(IMP orig) {
        return ^(id self, const void *b, NSUInteger len, NSUInteger idx) {
            sub(g_encBytes, self)[[@(idx) stringValue]] = @{@"len": @(len), @"hex": hexOf(b, len)};
            ((void (*)(id, SEL, const void *, NSUInteger, NSUInteger))orig)(self, @selector(setBytes:length:atIndex:), b, len, idx);
        };
    });
    hookAll(@selector(setBuffer:offset:atIndex:), ^id(IMP orig) {
        return ^(id self, id<MTLBuffer> b, NSUInteger off, NSUInteger idx) {
            sub(g_encBufs, self)[[@(idx) stringValue]] = @{@"buf": key(b), @"off": @(off), @"len": @(b ? b.length : 0), @"label": (b.label ?: @"")};
            ((void (*)(id, SEL, id, NSUInteger, NSUInteger))orig)(self, @selector(setBuffer:offset:atIndex:), b, off, idx);
        };
    });
    hookAll(@selector(setBuffers:offsets:withRange:), ^id(IMP orig) {
        return ^(id self, const id *bs, const NSUInteger *offs, NSRange r) {
            for (NSUInteger i = 0; i < r.length; i++) {
                id<MTLBuffer> b = bs[i];
                sub(g_encBufs, self)[[@(r.location + i) stringValue]] = @{@"buf": key(b), @"off": @(offs[i]), @"len": @(b ? b.length : 0), @"label": (b.label ?: @"")};
            }
            ((void (*)(id, SEL, const id *, const NSUInteger *, NSRange))orig)(self, @selector(setBuffers:offsets:withRange:), bs, offs, r);
        };
    });
    hookAll(@selector(setThreadgroupMemoryLength:atIndex:), ^id(IMP orig) {
        return ^(id self, NSUInteger len, NSUInteger idx) {
            sub(g_encTG, self)[[@(idx) stringValue]] = @(len);
            ((void (*)(id, SEL, NSUInteger, NSUInteger))orig)(self, @selector(setThreadgroupMemoryLength:atIndex:), len, idx);
        };
    });
    hookAll(@selector(dispatchThreadgroups:threadsPerThreadgroup:), ^id(IMP orig) {
        return ^(id self, MTLSize g, MTLSize t) {
            emitDispatch(self, @"groups", g, t, nil);
            ((void (*)(id, SEL, MTLSize, MTLSize))orig)(self, @selector(dispatchThreadgroups:threadsPerThreadgroup:), g, t);
        };
    });
    hookAll(@selector(dispatchThreads:threadsPerThreadgroup:), ^id(IMP orig) {
        return ^(id self, MTLSize g, MTLSize t) {
            emitDispatch(self, @"threads", g, t, nil);
            ((void (*)(id, SEL, MTLSize, MTLSize))orig)(self, @selector(dispatchThreads:threadsPerThreadgroup:), g, t);
        };
    });
    hookAll(@selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:), ^id(IMP orig) {
        return ^(id self, id<MTLBuffer> b, NSUInteger off, MTLSize t) {
            NSDictionary *extra = @{@"indirect": @{@"buf": key(b), @"off": @(off), @"hex": hexOf(b.contents ? (const char *)b.contents + off : NULL, 12)}};
            emitDispatch(self, @"indirect", MTLSizeMake(0, 0, 0), t, extra);
            ((void (*)(id, SEL, id, NSUInteger, MTLSize))orig)(self, @selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:), b, off, t);
        };
    });
    emit(@{@"ev": @"init", @"device": dev.name ?: @"?"});
}

void mbtrace_mark(const char *s) { emit(@{@"ev": @"mark", @"text": [NSString stringWithUTF8String:s ?: ""]}); }
