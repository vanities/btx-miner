#include <cstdint>
__global__ void k_tcgen05(){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 32;"::"l"(0ull)); }
__global__ void k_wgmma(){ asm volatile("wgmma.fence.sync.aligned;"); }
__global__ void k_tma(){ asm volatile("cp.async.bulk.commit_group;"); }
__global__ void k_cluster(){ asm volatile("barrier.cluster.arrive;"); }
__global__ void k_cpasync(){ asm volatile("cp.async.commit_group;"); }
__global__ void k_redux(){ int v=1,r; asm volatile("redux.sync.add.s32 %0,%1,0xffffffff;":"=r"(r):"r"(v)); }
int main(){ return 0; }
