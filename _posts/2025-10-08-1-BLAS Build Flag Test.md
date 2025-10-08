---
title: "[VulkanRT] 7. BLAS Build Flag Test"
categories:
  - Devs
tags: [Devs, Vulkan, RayTracing, BVH]
---

Keyword : BVH Refit, BVH Rebuild
![clustered_scene8]({{site.baseurl}}/assets/img/clustered_scene8.jpg)

## Description
참고 자료 : <a href="rtx-best-practices" target="_blank">https://developer.nvidia.com/blog/rtx-best-practices/</a>
1. Method A : `PREFER_FAST_BUILD`
	* Refit 허용 X, BVH Rebuild에 최적화 된 구조
	* ex) 파티클과 같이 지역 변형이 적고 크게 이동하는 경우
2. Method B : `PREFER_FAST_BUILD` | `ALLOW_UPDATE`
	* BVH Rebuild는 A보다 느리지만 Refit 허용
	* ex) Ray 교차할 확률이 비교적 적은 Low-LOD의 오브젝트
3. Method C : `PREFER_FAST_TRACE` | `ALLOW_UPDATE`
	* AS Update 옵션 중 가장 빠른 Trace 연산, 업데이트는 가장 느림.
	* ex) Ray 교차할 확률이 비교적 많은 High-LOD의 오브젝트

A,B,C를 Single BLAS Mesh, Cluster BLAS Mesh 두 BVH 방식에 대해 성능 테스트를 한다.


# Single BLAS Mesh vs Cluster BLAS Mesh A,B,C 테스트
![single_blas_mesh_and_cluster_blas_mesh]({{site.baseurl}}/assets/img/single_blas_mesh_and_cluster_blas_mesh.jpg)


* Single BLAS Mesh는 mesh를 하나의 BLAS로 만드는 기존 방식 (좌)
* Cluster BLAS Mesh는 mesh를 클러스터링 후 각 클러스터를 BLAS화 한 방식 (우)


### Single BLAS Mesh (8 Models)
<table>
  <thead>
    <tr>
      <th>항목</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BLAS Build Time</td>
      <td>0.681594 ms</td>
      <td>0.68151 ms</td>
      <td>0.825989 ms</td>
    </tr>
    <tr>
      <td>TLAS Build Time</td>
      <td>0.0135525 ms</td>
      <td>0.0134731 ms</td>
      <td>0.0138656 ms</td>
    </tr>
    <tr>
      <td>Total AS Build Time</td>
      <td>0.695147 ms</td>
      <td>0.694984 ms</td>
      <td>0.839855 ms</td>
    </tr>
    <tr>
      <td>Tracing Time</td>
      <td>0.573494 ms</td>
      <td>0.570867 ms</td>
      <td>0.570171 ms</td>
    </tr>
    <tr>
      <td>FPS</td>
      <td>145.803 fps (6.85856 ms)</td>
      <td>145.888 fps (6.85457 ms)</td>
      <td>143.497 fps (6.96878 ms)</td>
    </tr>
  </tbody>
</table>

<small>**Num BLASes**: **36**</small>

> Single BLAS 같은 경우 A,B는 사실상 같게 나왔고, C는 BLAS 빌드 시간이 증가했을 뿐 Tracing에서도 이점은 없었다.\
삼각형 개수가 더 많은 모델에 적용해야 이점이 있을 것으로 보인다.

### Cluster BLAS Mesh (8 Models)


<table>
  <thead>
    <tr>
      <th>항목</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BLAS Build Time</td>
      <td>0.380535 ms</td>
      <td>0.388019 ms</td>
      <td>9.20738 ms</td>
    </tr>
    <tr>
      <td>TLAS Build Time</td>
      <td>0.0235938 ms</td>
      <td>0.0236705 ms</td>
      <td>0.0238507 ms</td>
    </tr>
    <tr>
      <td>Total AS Build Time</td>
      <td>0.404129 ms</td>
      <td>0.41169 ms</td>
      <td>9.23123 ms</td>
    </tr>
    <tr>
      <td>Tracing Time</td>
      <td>0.694935 ms</td>
      <td>0.694623 ms</td>
      <td>0.55975 ms</td>
    </tr>
    <tr>
      <td>FPS</td>
      <td>148.291 fps (6.74348 ms)</td>
      <td>148.944 fps (6.71394 ms)</td>
      <td>62.7536 fps (15.9353 ms)</td>
    </tr>
  </tbody>
</table>

<small>**Num BLASes**: **2127**</small>

> A가 B보다 빌드 시간이 미세하게 빠른 것을 확인할 수 있고,\
C의 경우 삼각형의 개수는 같지만 리빌드해야 하는 BLAS 자체의 개수가 증가하여 BLAS 빌드 시간이 눈에 띄게 증가한 것을 볼 수 있다. 

<small>**Num Vertices**: **9,285**</small>\
<small>**Num Triangles**: **17,916**</small>\
<small>**Num Joints**: **168**</small>