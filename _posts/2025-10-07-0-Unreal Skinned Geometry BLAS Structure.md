---
title: "[Unreal 뜯어보기] Skinned Geometry BLAS 형태"
categories:
  - Finds
tags: [Finds, RayTracing, BVH, Skeletal Animation] # add tag
---

![Unreal Skinned Geometry BLAS 형태]({{site.baseurl}}/assets/img/unreal_skinned_geometry_blas.jpg)
<em>Nsight Graphics 캡쳐</em>

* Skinned Geometry당 하나의 BLAS
* 초록, 주황 부분이 각각 하위 Geoemtry
* Build Flag는 모두 Fast Build & Allow Update

움직임, 변형이 많은 어깨, 팔꿈치, 골반 부분에서 BLAS가 분리되는 것을 알 수 있다.
\+ bvh의 변형이 잦은 부분을 기준으로 파티셔닝을 한 것으로 예측했는데 그냥 skeletal mesh 자체의 세그먼트 단위로 나눈 거였다.
