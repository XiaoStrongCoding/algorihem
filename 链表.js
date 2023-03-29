/**
 * 链表 
 * /

/**
 * 2. 两数相加 https://leetcode.cn/problems/add-two-numbers/description/
 * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
    请你将两个数相加，并以相同形式返回一个表示和的链表。
    你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
 */

/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
 var addTwoNumbers = function(l1, l2) {
    let header = null
    let temp = null
    let addOne = 0
    while(l1 || l2) {
        const sum = (l1?l1.val:0) + (l2?l2.val:0) + addOne
        if(!header) {
            header = temp = new ListNode(sum % 10)
        } else {
            temp.next = new ListNode(sum % 10)
            temp = temp.next
        }
        
        addOne = Math.floor(sum / 10)

        l1 && (l1 = l1.next)
        l2 && (l2 = l2.next)
    }
    if(addOne) {
        temp.next = new ListNode(addOne)
    }
    return header
}

/**
* 19. 删除链表的倒数第 N 个结点 https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/
* 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
*/

/**
* @param {ListNode} head
* @param {number} n
* @return {ListNode}
*/
var removeNthFromEnd = function(head, n) {
   const res = new ListNode(0, head)
   let behind = front = res
   while(n--) {
       front = front.next
   }

   while(front.next) {
       front = front.next
       behind = behind.next
   }

   behind.next = behind.next.next

   return res.next
};

/**
* 21. 合并两个有序链表 https://leetcode.cn/problems/merge-two-sorted-lists/description/
* 将两个升序链表合并为一个新的 升序 链表并返回。
*/

/**
* @param {ListNode} list1 方法一
* @param {ListNode} list2
* @return {ListNode}
*/
var mergeTwoLists = function(list1, list2) {
   if((!list1 && !list2) || (!list1 && list2)) return list2

   if(list1&&!list2) return list1

   if(list1.val > list2.val) {
       list2.next = mergeTwoLists(list1, list2.next)
       return list2
   } else {
       list1.next = mergeTwoLists(list1.next, list2)
       return list1
   }
};

/**
* @param {ListNode} list1
* @param {ListNode} list2
* @return {ListNode} 方法二
*/
var mergeTwoLists = function(list1, list2) {const header = new ListNode(-1)
   let pre = header
   while(list1 && list2) {
       if(list1.val <= list2.val) {
           pre.next = list1
           list1 = list1.next
       } else {
           pre.next = list2
           list2 = list2.next
       }
       pre = pre.next
   }
   pre.next = list1 ? list1 : list2
   return header.next
};

/**
* 23. 合并K个升序链表 https://leetcode.cn/problems/merge-k-sorted-lists/description/
   给你一个链表数组，每个链表都已经按升序排列。
   请你将所有链表合并到一个升序链表中，返回合并后的链表。
*/

/**
* @param {ListNode[]} lists 方法一
* @return {ListNode}
*/
var mergeKLists = function(lists) {
   if(!lists.length) return null

   const res = new ListNode(-1)
   let temp = res

   lists = lists.filter(Boolean)
   while(lists.length > 1) {
       let minIndex = 0
       for(let i=1; i<lists.length; i++) {
           if(lists[minIndex].val > lists[i].val) {
               minIndex = i
           }
       }
       temp.next = lists[minIndex]
       temp = temp.next
       lists[minIndex] = lists[minIndex].next
       lists = lists.filter(Boolean)
   }

   temp.next = lists.length ? lists[0] : null

   return res.next
};

/**
* @param {ListNode[]} lists 方法二
* @return {ListNode}
*/
var mergeKLists = function(lists) {
   return lists.reduce((res, list)=>{
       while(list) {
           res.push(list)
           list = list.next
       }
       return res
   },[]).sort((a,b) => a.val - b.val).reduceRight((p, node)=>(node.next=p, p = node, p), null)
};

/**
 * 148. 排序链表 https://leetcode.cn/problems/sort-list/description/
 * 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
 */

/**
 * @param {ListNode} head
 * @return {ListNode}
 */
 var sortList = function (head) {
    if(!head || !head.next) return head

    let low = fast = head
    let pre = null
    while(fast && fast.next) {
        pre = low
        low = low.next
        fast = fast.next.next
    }
    pre.next = null
    const l = sortList(head)
    const r = sortList(low)
    return merge(l, r)
};

function merge(l, r) {
    const pre = temp = new ListNode()

    while(l&&r) {
        if(l.val>r.val) {
            temp.next = r
            r = r.next
        } else {
            temp.next = l
            l = l.next
        }
        temp = temp.next
    }

    l && (temp.next = l)
    r && (temp.next = r)

    return pre.next
}

/*********************循环链表 start**********************/

/**
 * 141. 环形链表 https://leetcode.cn/problems/linked-list-cycle/description/
 * 给你一个链表的头节点 head ，判断链表中是否有环。
 */

/**
 * @param {ListNode} head
 * @return {boolean}
 */
var hasCycle = function(head) {
    while(head) {
        if(head.tag) return true

        head.tag = true
        head = head.next
    }
    return false
};

/**
 * 142. 环形链表 II https://leetcode.cn/problems/linked-list-cycle-ii/description/?favorite=2cktkvj
 * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
 */

/**
 * @param {ListNode} head
 * @return {ListNode}
 */
 var detectCycle = function(head) {
    while(head) {
        if(head.tag) return head
        head.tag = true
        head = head.next
    }
    return null
};

/**
 * 龟兔赛跑
 * 287. 寻找重复数 https://leetcode.cn/problems/find-the-duplicate-number/description/?favorite=2cktkvj
 * 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
    假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。
    你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var findDuplicate = function(nums) {
    let low = 0, fast = 0

    do{
        low = nums[low];
        fast = nums[nums[fast]]
     } while (low !== fast)

     low = 0
     while(low !== fast) {
         low = nums[low]
         fast = nums[fast]
     }

     return fast
};

/*********************循环链表 end**********************/

// 双向链表
/**
 * 146. LRU 缓存 https://leetcode.cn/problems/lru-cache/description/
 * 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
    实现 LRUCache 类：
    LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
    int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
    void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；
        如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，
        则应该 逐出 最久未使用的关键字。
 */

 var TreeNode = function(key, val) {
    this.key = key
    this.val = val
    this.next = null
    this.pre = null
}

/**
 * @param {number} capacity
 */
var LRUCache = function(capacity) {
    this.capacity = capacity
    this.count = 0
    this.cache = {}
    this.nodeHead = new TreeNode()
    this.nodeTail = new TreeNode()
    this.nodeHead.next = this.nodeTail
    this.nodeTail.pre = this.nodeHead

    this.removeNodeFromList = (node) => {
        node.pre.next = node.next
        node.next.pre = node.pre
    }

    this.addNodeToHead = (node) => {
        node.next = this.nodeHead.next
        this.nodeHead.next.pre = node
        this.nodeHead.next = node
        node.pre = this.nodeHead
    }
};

/** 
 * @param {number} key
 * @return {number}
 */
LRUCache.prototype.get = function(key) {
    const node = this.cache[key]
    if(!node) return -1
    this.removeNodeFromList(node)
    this.addNodeToHead(node)
    return node.val
};

/** 
 * @param {number} key 
 * @param {number} value
 * @return {void}
 */
LRUCache.prototype.put = function(key, value) {
    const cacheNode = this.cache[key]

    if(!cacheNode) {
        const node = new TreeNode(key, value)
        if(this.count === this.capacity) {
            const removeNode = this.nodeTail.pre
            this.count--
            delete this.cache[removeNode.key]
            this.removeNodeFromList(removeNode)
        }

        this.count++
        this.cache[key] = node
        this.addNodeToHead(node)
    } else {
        this.removeNodeFromList(cacheNode)
        this.addNodeToHead(cacheNode)
        cacheNode.val = value
    }
};

/**
 * 160. 相交链表 https://leetcode.cn/problems/intersection-of-two-linked-lists/description/
 * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
 */

/**
 * @param {ListNode} headA
 * @param {ListNode} headB
 * @return {ListNode}
 */
 var getIntersectionNode = function(headA, headB) {
    while(headA || headB) {
        if(headA === headB) return headA
        if(headA && headA.tag) return headA
        if(headB && headB.tag) return headB

        if(headA) {
            headA.tag = true
            headA = headA.next
        }

        if(headB) {
            headB.tag = true
            headB = headB.next
        }
    }

    return null
};

/**
 * 206. 反转链表 https://leetcode.cn/problems/reverse-linked-list/description/
 * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
 */

/**
 * @param {ListNode} head
 * @return {ListNode}
 */
 var reverseList = function(head) {
    let node
    function traverse(pre, nex) {
        if(nex == null) {
            node = pre
            return
        }

        const temp = nex.next
        nex.next = pre
        traverse(nex, temp)
    }

    traverse(null, head)

    return node
};

var reverseList = function(head) {
    if(!head || !head.next) return head
    let pre = null, cur = head
    while(cur) {
        const temp = cur.next
        cur.next = pre
        pre = cur
        cur = temp
    }
    return pre
};

/**
 * 234. 回文链表 https://leetcode.cn/problems/palindrome-linked-list/description/
 * 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。
 */

/**
 * @param {ListNode} head
 * @return {boolean}
 */
 var isPalindrome = function(head) {
    const all = []
    while(head) {
        all.push(head.val)
        head = head.next
    }

    let l =0, r = all.length-1

    while(l<=r) {
        if(all[l]!==all[r]) return false
        l++;
        r--;
    }

    return true
};