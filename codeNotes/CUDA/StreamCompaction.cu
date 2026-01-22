#define THRUST 1
#define CUB 1

// Thrust
#if(THRUST)
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// 1. 오프셋 계산용 Functor (인덱스 -> 인덱스 * 128)
struct StrideOp {
    __host__ __device__
    uint32_t operator()(uint32_t idx) const {
        return idx * 128; // 128 stride
    }
};

// 2. 유효성 검사 Functor
struct IsValid {
    __host__ __device__
    bool operator()(uint32_t val) const {
        return val != 0xffffffff;
    }
};

void CompactDataAndOffsets(
    uint32_t* d_inData,      // 입력 데이터
    uint32_t* d_outData,     // 결과 데이터가 저장될 곳
    uint32_t* d_outOffsets,  // 결과 오프셋이 저장될 곳
    int N,                   // 전체 개수
    int* h_outCount)         // (선택) 살아남은 개수 반환
    {
        // A. 가상의 오프셋 반복자 생성 (메모리 할당 X, 즉석 계산)
        // 0, 1, 2, 3...
        auto counting = thrust::make_counting_iterator(0);
        // 0, 128, 256, 384...
        auto strided_offsets = thrust::make_transform_iterator(counting, StrideOp());
        
        // B. 입력과 오프셋을 하나로 묶음 (Zip)
        // Input: [(Data[0], 0), (Data[1], 128), (Data[2], 256)...]
        auto zip_input = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<uint32_t>(d_inData), 
            strided_offsets
        ));
        
        // C. 출력도 하나로 묶음
        auto zip_output = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<uint32_t>(d_outData), 
            thrust::device_ptr<uint32_t>(d_outOffsets)
        ));
        
        // D. copy_if 실행 (Stencil 사용)
        // zip_input: 복사할 대상 (데이터 + 오프셋)
        // d_inData: 조건을 검사할 대상 (데이터가 유효한지 확인)
        // IsValid: 조건 함수
        auto result_end = thrust::copy_if(
            zip_input,              // Source (무엇을 옮길 것인가)
            zip_input + N,          // End
            thrust::device_ptr<uint32_t>(d_inData), // Stencil (조건 검사 대상)
            zip_output,             // Destination
            IsValid()               // Predicate
        );
        
        // (선택) 결과 개수 계산
        if (h_outCount) {
            *h_outCount = result_end - zip_output;
        }
    }
    
#endif

// CUB
#if(CUB)
#include <cub/cub.cuh>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

// 오프셋 계산용 (인덱스 -> 인덱스 * 128)
struct StrideOp {
    __host__ __device__ __forceinline__
    uint32_t operator()(uint32_t idx) const {
        return idx * 128;
    }
};

// 튜플(값, 오프셋)에서 값을 확인해 필터링하는 Functor
struct FilterOp {
    __host__ __device__ __forceinline__
    bool operator()(const thrust::tuple<uint32_t, uint32_t>& item) const {
        // tuple의 0번째 요소(값)가 0xffffffff가 아니면 true(남김)
        return thrust::get<0>(item) != 0xffffffff;
    }
};

// [초기화 단계] 한 번만 할당하고 재사용할 임시 메모리 래퍼
struct CubTempStorage {
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    
    ~CubTempStorage() { if(d_temp) cudaFree(d_temp); }
    
    void EnsureCapacity(size_t needed) {
        if (temp_bytes < needed) {
            if (d_temp) cudaFree(d_temp);
            cudaMalloc(&d_temp, needed);
            temp_bytes = needed;
        }
    }
};

// [실행 함수]
void CompactWithOffsetCUB(
    uint32_t* d_inData,      // 입력
    uint32_t* d_outData,     // 결과 값
    uint32_t* d_outOffsets,  // 결과 오프셋
    int* d_numSelected,      // 결과 개수 (Device 포인터여야 함)
    int numItems,            // 전체 개수
    CubTempStorage& storage) // 미리 만들어둔 스토리지
{
    // 1. Input Iterator 구성 (Data + Calculated Offset)
    //    메모리에서 Data만 읽고, Offset은 레지스터에서 즉시 계산 (Load 대역폭 절약)
    auto count_iter = thrust::make_counting_iterator(0);
    auto stride_iter = thrust::make_transform_iterator(count_iter, StrideOp());
    
    auto input_zip = thrust::make_zip_iterator(thrust::make_tuple(
        d_inData,    // 0: 값
        stride_iter  // 1: 오프셋 (0, 128, 256...)
    ));

    // 2. Output Iterator 구성
    auto output_zip = thrust::make_zip_iterator(thrust::make_tuple(
        d_outData,
        d_outOffsets
    ));

    // 3. CUB DeviceSelect 실행
    //    첫 호출시(d_temp == nullptr)에는 필요한 크기만 계산하여 리턴
    if (storage.d_temp == nullptr) {
        cub::DeviceSelect::If(nullptr, storage.temp_bytes, input_zip, output_zip, d_numSelected, numItems, FilterOp());
        storage.EnsureCapacity(storage.temp_bytes);
    }

    //    실제 수행
    cub::DeviceSelect::If(storage.d_temp, storage.temp_bytes, input_zip, output_zip, d_numSelected, numItems, FilterOp());
}
#endif