#include <cub/cub.cuh>
#include <iostream>
#include <vector>

// 조건: 양수만 선택
struct IsPositive {
    __device__ bool operator()(int x) const {
        return x > 0;
    }
};

int main() {
    const int N = 8;
    int h_in[N] = { -3, 0, 5, -1, 7, 2, -8, 9 };

    // Device 메모리 할당
    int *d_in, *d_out, *d_num_selected;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMalloc(&d_num_selected, sizeof(int));

    // 입력 데이터 복사
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // temp storage 준비
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 1단계: 필요한 임시 버퍼 크기 계산
    cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes,
        d_in, d_out, d_num_selected,
        N, IsPositive()
    );

    // 2단계: 실제 연산 수행
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes,
        d_in, d_out, d_num_selected,
        N, IsPositive()
    );

    // 결과 개수 가져오기
    int h_num_selected;
    cudaMemcpy(&h_num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);

    // 결과 배열 가져오기
    std::vector<int> h_out(h_num_selected);
    cudaMemcpy(h_out.data(), d_out, h_num_selected * sizeof(int), cudaMemcpyDeviceToHost);

    // 출력
    std::cout << "Selected count: " << h_num_selected << std::endl;
    std::cout << "Selected values: ";
    for (int v : h_out) std::cout << v << " ";
    std::cout << std::endl;

    // 메모리 해제
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_num_selected);
    cudaFree(d_temp_storage);

    return 0;
}