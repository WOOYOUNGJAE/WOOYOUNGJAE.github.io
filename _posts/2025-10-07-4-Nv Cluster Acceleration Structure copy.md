---
title: "[VulkanRT] 3. Nv Cluster Acceleration Structure"
categories:
  - Devs
tags: [Devs, Vulkan, RayTracing, CLAS, BVH]
---
Keyword : Cluster Acceleration Structure


![cluster_acceleration_structure]({{site.baseurl}}/assets/img/cluster_acceleration_structure.jpg)



## Description
Nvidia의 Cluster Acceleration Structure Extension\
단순 적용만 해보며 겪은 문제만 간단하게 정리해보았다.\
기존 Traditional BLAS와의 성능 비교 등 실험은 이후 포스팅에서 다룰 것이다.


## CLAS Transform
### 기존의 Raytracing 같은 경우
Node의 Transform을
1. vertex position에 선반영할지
2. acceleration structure을 빌드할 때 input transform에 입력할지

두 가지 옵션이 있었다. 그러나

### CLAS 로 구성되는 Clustered BLAS 같은 경우
Geometry의 node 계층을 고려하지 않고 임의의 cluster로 분할하였기 때문에 어떤 cluster에 어떤 node matrix를 적용할지 알 수 없다.\
더군다나 cluster build input 구조체 역시 trasnform 입력 변수가 없다.
```c++
typedef struct VkClusterAccelerationStructureBuildTriangleClusterInfoNV {
    uint32_t                                                         clusterID;
    VkClusterAccelerationStructureClusterFlagsNV                     clusterFlags;
    uint32_t                                                         triangleCount:9;
    uint32_t                                                         vertexCount:9;
    uint32_t                                                         positionTruncateBitCount:6;
    uint32_t                                                         indexType:4;
    uint32_t                                                         opacityMicromapIndexType:4;
    VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV    baseGeometryIndexAndGeometryFlags;
    uint16_t                                                         indexBufferStride;
    uint16_t                                                         vertexBufferStride;
    uint16_t                                                         geometryIndexAndFlagsBufferStride;
    uint16_t                                                         opacityMicromapIndexBufferStride;
    VkDeviceAddress                                                  indexBuffer;
    VkDeviceAddress                                                  vertexBuffer;
    VkDeviceAddress                                                  geometryIndexAndFlagsBuffer;
    VkDeviceAddress                                                  opacityMicromapArray;
    VkDeviceAddress                                                  opacityMicromapIndexBuffer;
} VkClusterAccelerationStructureBuildTriangleClusterInfoNV;
```
따라서 각 vertex에 gltf node matrix를 선반영하는 것이 불가피하다.