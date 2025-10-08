---
title: "[VulkanRT] 2. Skeletal Mesh Animation Raytracing"
categories:
  - Devs
tags: [Devs, Vulkan, RayTracing, Animation]
---
Keywords : skeletal mesh, skinning, animation, compute shader

![basic_skeletal_animation]({{site.baseurl}}/assets/img/skeletal_animation_rt_8.jpg)

Codes: [MyDevs/myRayTracingLittleAdvanced/mySkeletalAnimationRT.cpp](https://github.com/WOOYOUNGJAE/VulkanMyDevs/blob/master/MyDevs/myRayTracingLittleAdvanced/mySkeletalAnimationRT.cpp)


# Description
## compute skinning [(anim.comp)](https://github.com/WOOYOUNGJAE/VulkanMyDevs/blob/master/shaders/glsl/myRayTracingLittleAdvanced/anim.comp)
compute animation은 실제 vertex data에 write를 하기 때문에 최초의 상태(T pose)가 유지되어야 한다.\
따라서 다음 두 가지 vertex buffer을 사용한다.
1. compute shader의 input으로 활용할 "T pose Vertex Buffer"
2. compute shader의 output으로 활용할 "Deforming Vertex Buffer"

이후 변환된 Deforming Vertex Buffer을 acceleration sturcture build의 input에 입력한다.

```c++
if (isDeformable)
	vertexBuffer = model.deformingVertices.buffer;
else
	vertexBuffer = model.vertices.buffer;
// ....
asGeometry.geometry.triangles.vertexData = getBufferDeviceAddress(vertexBuffer);
```
---
# Sascha의 Matrix Update 개선
```c++
void Node::update()
{
	if (mesh) {
		glm::mat4 m = getMatrix(); // from current to root
		if (skin) {
			mesh->uniformBlock.matrix = m;
			// Update joint matrices
			glm::mat4 inverseTransform = glm::inverse(m);
			for (size_t i = 0; i < skin->joints.size(); i++) {
				myglTF::Node* jointNode = skin->joints[i];
				glm::mat4 jointMat = jointNode->getMatrix() * skin->inverseBindMatrices[i];
				jointMat = inverseTransform * jointMat;
				mesh->uniformBlock.jointMatrix[i] = jointMat;
			}
			//..
		}
		//..
	}
	for (auto& child : children)
		child->update();
}
```
- 위 코드는 Sacha의 Node Update 코드인데 두 가지 문제가 있다.

## 문제 1 : Tree 구조의 이점을 살리지 못한 Joint Node 업데이트
joint(bone) 업데이트는 보통 Top-Down 방식으로 부모 노드의 Matrix에 현재 노드의 Matrix를 곱하여 누적함으로써 자식에서 반복을 피한다.\
그러나 Sascha 코드의 경우 모든 Joint들을 순회하며 부모 노드까지 Down-Top 방향으로 Matrix를 업데이트한다.\
updateJoints() 함수를 추가하여 이를 개선하였다.

```c++
void updateJoints(glm::mat4 parentMatrix, std::array<glm::mat4, MAX_JOINTS>& jointMatrices)
{
	//.. 	
	glm::mat4 curNodeMatrix = localMatrix();
	glm::mat4 toRoot = parentMatrix * curNodeMatrix;

	// curjointSpace -> jointRoot
	jointMatrices[jointIndexInSkin] = toRoot;

	for (auto& child : children)
		child->updateJoints(toRoot, jointMatrices);
}
```
## 문제 2 : 모든 Mesh에 대해 JointMatrix 업데이트

gltf 모델은 다수의 Mesh가 하나의 Skin을 공유할 수 있다.\
따라서 여러 Mesh들이 공유하는 Skin의 Joint들을 1회 업데이트 후 이것을 활용하면 되는데\
Sascha 코드의 경우 Skin을 가지는 Mesh에 대해 모든 Joint Matrix를 업데이트 하도록 되어 있다.\
이를 Skin을 가지는 Mesh의 경우 Skin 고유의 JointMatrix 배열을 찾아서 활용하도록 하였다.

```c++
void myglTF::ModelRT::updateNodeTransforms(Node* pNode)
{
	if (pNode->mesh) {
		glm::mat4 m = pNode->getMatrix();
		if (pNode->skin) {
			pNode->mesh->uniformBlock.matrix = m;

			const auto& jointMatrices = rootToMatricesMap[pNode->skin->jointRoot];
			//..
			for (size_t i = 0; i < pNode->skin->joints.size(); i++) {
				myglTF::Node* jointNode = pNode->skin->joints[i];
				glm::mat4 jointMat = jointMatrices[i] * pNode->skin->inverseBindMatrices[i];
				jointMat = inverseTransform * jointMat;
				pNode->mesh->uniformBlock.jointMatrix[i] = jointMat;
				//..
			}
	} //..
}
```

### 성능 향상
<table>
  <thead>
    <tr>
      <th>항목</th>
      <th>개선 전(Sascha)</th>
      <th>개선 후</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Animation CPU Time</strong></td>
      <td>4.05584 (ms)</td>
      <td>0.87191 (ms)<br><sub>78.54% 성능 향상</sub></td>
    </tr>
    <tr>
      <td><strong>FPS</strong></td>
      <td>94.422 fps (10.5908 ms)</td>
      <td>132.076fps (7.5714 ms)</td>
    </tr>
  </tbody>
</table>

<small>**Num Vertices**: **126,150**</small><br/>
<small>**Num Triangles**: **234,277**</small><br/>
<small>**Num Joints**: **640**</small><br/>
<small>**Measured Frame Count**: **1000**</small>