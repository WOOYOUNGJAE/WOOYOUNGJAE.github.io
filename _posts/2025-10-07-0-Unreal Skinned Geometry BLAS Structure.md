---
title: "[RT/SkeletalAnim] Unreal Skinned Geometry BLAS 형태"
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