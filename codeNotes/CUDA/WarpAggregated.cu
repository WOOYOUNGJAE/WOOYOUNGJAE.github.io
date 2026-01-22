__global__ void compactTriangles(..., int* clusterCounters, Triangle* outputBuffer, int* clusterOffsets) {
    // ... (앞부분 생략: tid 계산 및 clusterID 찾기) ...
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int clusterID = getClusterID(tid); // 각자 자신이 속한 클러스터 ID 계산

    // ---------------------------------------------------------
    // 1. Warp 내에서 "나와 같은 클러스터 ID"를 가진 쓰레드 찾기
    // ---------------------------------------------------------
    unsigned int active = __activemask();
    // __match_any_sync: active 마스크 내에서 같은 clusterID를 가진 쓰레드들의 마스크 반환
    unsigned int peers = __match_any_sync(active, clusterID);

    // 2. 그룹 내에서 나의 순서(Rank) 계산 (몇 번째인지)
    // lane_mask: 현재 쓰레드(laneId)보다 앞선 비트들만 1인 마스크
    unsigned int lane_mask = (1 << (threadIdx.x & 31)) - 1;
    // 내 앞의 동료 수 = 나의 로컬 오프셋
    int local_rank = __popc(peers & lane_mask);

    // 3. 그룹의 리더(가장 앞선 쓰레드) 선출 및 총 개수 계산
    // __ffs: Find First Set (1-based index), 리더의 Lane ID를 찾음
    int leader_lane = __ffs(peers) - 1;
    int group_size = __popc(peers); // 이 그룹의 총 인원 수

    int base_offset = 0;

    // 4. 리더만 대표로 전역 메모리에 Atomic Add 수행
    if (threadIdx.x % 32 == leader_lane) {
        // clusterCounters는 커널 실행 전 0으로 초기화 필수
        base_offset = atomicAdd(&clusterCounters[clusterID], group_size);
    }

    // 5. 리더가 받아온 시작 주소를 그룹원들에게 공유 (Broadcast)
    base_offset = __shfl_sync(peers, base_offset, leader_lane);

    // ---------------------------------------------------------
    // 6. 최종 쓰기 위치 계산 및 데이터 저장
    // ---------------------------------------------------------
    int final_index = base_offset + local_rank;
    
    // 출력 버퍼의 해당 클러스터 영역에 저장
    // (예: clusterOffsets[clusterID]가 해당 클러스터의 시작 지점이라고 가정)
    int global_write_pos = clusterOffsets[clusterID] + final_index;
    
    outputBuffer[global_write_pos] = inputTriangles[tid];
}