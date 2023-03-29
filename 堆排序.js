/**
 * 堆排序 https://leetcode.cn/problems/kth-largest-element-in-an-array/solutions/836425/xie-gei-qian-duan-tong-xue-de-ti-jie-yi-kt5p2/
 * 用处：不一定要全部排序好再去拿，只针对部分元素进行排序
 * 原理：上浮下沉
 * 时间复杂度均为 O(nlogn)
 * 堆是具有以下性质的完全二叉树：
    大顶堆：每个节点的值都 大于或等于 其左右孩子节点的值
    注：没有要求左右值的大小关系

    小顶堆：每个节点的值都 小于或等于 其左右孩子节点的值

   步骤：
    第一步构建初始堆：是自底向上构建，从最后一个非叶子节点开始。
    第二步就是下沉操作让尾部元素与堆顶元素交换，最大值被放在数组末尾，并且缩小数组的length，不参与后面大顶堆的调整
    第三步就是调整：是从上到下，从左到右,因为堆顶元素下沉到末尾了，要重新调整这颗大顶堆
 */

/**
 * 215. 数组中的第K个最大元素 https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?favorite=2cktkvj
 * 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
    请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
    你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
 */

/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
 var findKthLargest = function(nums, k) {
    let heapSize=nums.length
    buildMaxHeap(nums,heapSize) // 构建好了一个大顶堆
    // 进行下沉 大顶堆是最大元素下沉到末尾
    for(let i=nums.length-1;i>=nums.length-k+1;i--){
        swap(nums,0,i)
        --heapSize // 下沉后的元素不参与到大顶堆的调整
        // 重新调整大顶堆
         maxHeapify(nums, 0, heapSize);
    }
    return nums[0]
   // 自下而上构建一颗大顶堆
   function buildMaxHeap(nums,heapSize){
     for(let i=Math.floor(heapSize/2)-1;i>=0;i--){
        maxHeapify(nums,i,heapSize)
     }
   }
   // 从左向右，自上而下的调整节点
   function maxHeapify(nums,i,heapSize){
       let l=i*2+1
       let r=i*2+2
       let largest=i
       if(l < heapSize && nums[l] > nums[largest]){
           largest=l
       }
       if(r < heapSize && nums[r] > nums[largest]){
           largest=r
       }
       if(largest!==i){
           swap(nums,i,largest) // 进行节点调整
           // 继续调整下面的非叶子节点
           maxHeapify(nums,largest,heapSize)
       }
   }
   function swap(a,  i,  j){
        let temp = a[i];
        a[i] = a[j];
        a[j] = temp;
   }
};