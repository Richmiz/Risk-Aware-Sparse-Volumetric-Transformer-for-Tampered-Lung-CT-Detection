"""Measured timing/memory and explicitly scoped analytic operator FLOP estimates."""

from contextlib import contextmanager
import math
import platform
from pathlib import Path
import time
import torch


def synchronize(device):
    if torch.device(device).type=="cuda":torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self,device,enabled=False):self.device=device;self.enabled=enabled;self.ms={}
    @contextmanager
    def stage(self,name):
        if not self.enabled:yield;return
        synchronize(self.device);start=time.perf_counter()
        yield
        synchronize(self.device)
        self.ms[name]=self.ms.get(name,0.)+(time.perf_counter()-start)*1000


def percentile(values,q):
    if not values or not all(math.isfinite(v) and v>=0 for v in values):raise ValueError("Finite nonnegative samples required")
    values=sorted(values);position=(len(values)-1)*q;left=math.floor(position);right=math.ceil(position)
    return values[left]+(values[right]-values[left])*(position-left)


def process_memory():
    """Fresh-process high-water RSS includes imports, model, inputs and allocator."""
    if platform.system()=="Linux":
        values={}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):
                key,value,*_=line.split();values[key.rstrip(':')]=int(value)*1024
        return {"rss_bytes":values.get('VmRSS'),"peak_rss_bytes":values.get('VmHWM'),"scope":"process_including_imports_model_inputs_allocator"}
    if platform.system()=="Windows":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                *[(key,ctypes.c_size_t) for key in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        info=Counters();info.cb=ctypes.sizeof(info)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
            return {"rss_bytes":info.WorkingSetSize,"peak_rss_bytes":info.PeakWorkingSetSize,
                    "scope":"process_including_imports_model_inputs_allocator"}
    return {"rss_bytes":None,"peak_rss_bytes":None,"scope":"unavailable"}


def estimated_operator_flops(model,shapes,constructed_counts,sequence_length,*,router_used):
    """2 FLOPs/MAC; executed padded Conv3d, Linear, QKV/out, QK/AV and FFN only."""
    d=model.classifier.linear.in_features;w=model.coarse_encoder.first.out_channels
    c=model.tokenizer.first.out_channels;coarse=candidate=0
    for shape,counts in zip(shapes,constructed_counts):
        padded=[math.ceil(n/model.coarse_patch_size)*model.coarse_patch_size for n in shape]
        first=math.prod(n//2 for n in padded);second=math.prod(n//4 for n in padded)
        coarse+=first*w*27+second*d*w*27+counts[0]*d*d
        p=max(model.patch_sizes);padded=[math.ceil(n/p)*p for n in shape]
        candidate+=math.prod(padded)*c*5*27+math.prod(n//2 for n in padded)*d*c*27+sum(counts[1:])*d*d
    batch=len(shapes);raw=batch*max(sum(count) for count in constructed_counts)
    position=raw*(12*d+d*d)
    hidden=model.router.score[0].out_features
    router=raw*((2*d+5)*hidden+hidden) if router_used else 0
    transformer=batch*model.layers*(12*sequence_length*d*d+2*sequence_length**2*d)
    values={"coarse_encoder":coarse*2,"candidate_encoder_and_projections":candidate*2,"physical_position":position*2,
            "controller":router*2,"transformer":transformer*2,"classification":batch*d*4*2}
    return {"by_component":values,"total":sum(values.values()),"kind":"analytic_executed_operator_estimate_2_FLOPs_per_MAC",
        "excluded":"pooling,cues,norm,activation,softmax,bias,sorting,gather,validation,metadata,memory_traffic",
        "not_measured_hardware_FLOPs":True}
