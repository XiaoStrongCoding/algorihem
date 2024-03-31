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

// 方法二
var getIntersectionNode = function(headA, headB) {
    if(!headA || !headB) return null

    let h1 = headA, h2 = headB
    while(h1!==h2) {
        h1 = !h1 ? headB : h1.next
        h2 = !h2 ? headA : h2.next
    }

    return h1
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
 * 链表内指定区间反转 https://www.nowcoder.com/practice/b58434e200a648c589ca2063f1faf58c?tpId=295&tqId=654&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj
 * 将一个节点数为 size 链表 m 位置到 n 位置之间的区间反转，要求时间复杂度 O(n)，空间复杂度 O(1)。
例如：
给出的链表为 
1→2→3→4→5→NULL, 
m=2,n=4,
返回 
1→4→3→2→5→NULL.
 */

// 题解：1. 先找出要反转区域的钩子，A -> B -> C -> D, 当B,C为反转区域时钩子为A,D;       2. m是否等于1分情况;       3. 反转链表;
function reverseBetween( head ,  m ,  n ) {
    // write code here
    if(m===n || !head || !head.next) return head

    let cur = head
    let start = null
    let end = null
    for(let i=1; i<=n; i++) {
        if(i==m-1) {
            start = cur
        }
        cur = cur.next
    }
    end = cur

    let pre = null
    let temp = null

    if(m>1) {
        cur = start.next
        while(cur !== end) {
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        }
        start.next.next = cur
        start.next = pre
    } else {
        cur = head
        while(cur !== end) {
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        }
        head.next = cur
        head = pre
    }

    return head
}

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

const reverseList = (head) => {
    let prev = null;
    let curr = head;
    while (curr !== null) {
        let nextTemp = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextTemp;
    }
    return prev;
}

const endOfFirstHalf = (head) => {
    let fast = head;
    let slow = head;
    while (fast.next !== null && fast.next.next !== null) {
        fast = fast.next.next;
        slow = slow.next;
    }
    return slow;
}

var isPalindrome = function(head) {
    if (head == null) return true;

    // 找到前半部分链表的尾节点并反转后半部分链表
    const firstHalfEnd = endOfFirstHalf(head);
    const secondHalfStart = reverseList(firstHalfEnd.next);

    // 判断是否回文
    let p1 = head;
    let p2 = secondHalfStart;
    let result = true;
    while (result && p2 != null) {
        if (p1.val != p2.val) result = false;
        p1 = p1.next;
        p2 = p2.next;
    }        

    // 还原链表并返回结果
    firstHalfEnd.next = reverseList(secondHalfStart);
    return result;
};

/**
 * 138. 随机链表的复制 https://leetcode.cn/problems/copy-list-with-random-pointer/description/
 * 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
 */

/**
 * @param {Node} head
 * @return {Node}
 */
 var copyRandomList = function(head, nodeCache=new WeakMap()) {
    if(!head) return null
    if(!nodeCache.has(head)) {
        nodeCache.set(head, {val: head.val})
        Object.assign(nodeCache.get(head), {next: copyRandomList(head.next, nodeCache), random: copyRandomList(head.random, nodeCache)})
    }
    return nodeCache.get(head)
};

/**
 * 24.两两交换链表中的节点 https://leetcode.cn/problems/swap-nodes-in-pairs/description/
 * 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
 */

/**
 * @param {ListNode} head
 * @return {ListNode}
 */
 var swapPairs = function(head) {
    if(!head || !head.next) return head

    const newHead = head.next
    head.next = swapPairs(newHead.next)
    newHead.next = head
    return newHead
};

/**
 * 23. 合并 K 个升序链表 https://leetcode.cn/problems/merge-k-sorted-lists/description/
 * 给你一个链表数组，每个链表都已经按升序排列。
   请你将所有链表合并到一个升序链表中，返回合并后的链表。
 */

/**
 * @param {ListNode[]} lists
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

// 两两合并
var mergeTwoLists = function (list1, list2) {
    let dummy = new ListNode(); // 用哨兵节点简化代码逻辑
    let cur = dummy; // cur 指向新链表的末尾
    while (list1 && list2) {
        if (list1.val < list2.val) {
            cur.next = list1; // 把 list1 加到新链表中
            list1 = list1.next;
        } else { // 注：相等的情况加哪个节点都是可以的
            cur.next = list2; // 把 list2 加到新链表中
            list2 = list2.next;
        }
        cur = cur.next;
    }
    cur.next = list1 ? list1 : list2; // 拼接剩余链表
    return dummy.next;
};

var mergeKLists = function (lists) {
    // 合并从 lists[i] 到 lists[j-1] 的链表
    function dfs(i, j) {
        const m = j - i;
        if (m === 0) return null; // 注意输入的 lists 可能是空的
        if (m === 1) return lists[i]; // 无需合并，直接返回
        const left = dfs(i, i + (m >> 1)); // 合并左半部分
        const right = dfs(i + (m >> 1), j); // 合并右半部分
        return mergeTwoLists(left, right); // 最后把左半和右半合并
    }
    return dfs(0, lists.length);
};

/**
 * 25. K 个一组翻转链表 https://leetcode.cn/problems/reverse-nodes-in-k-group/description/?envType=study-plan-v2&envId=top-100-liked
 * 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
 * 输入：head = [1,2,3,4,5], k = 2
   输出：[2,1,4,3,5]
 */

/**
 * @param {ListNode} head
 * @param {number} k
 * @return {ListNode}
 */
 var reverseKGroup = function(head, k) {
    const hair = new ListNode(0)
    hair.next = head
    let pre = hair

    while(head) {
        let tail = pre
        for(let i=0; i<k; i++) {
            tail = tail.next
            if(!tail) {
                return hair.next
            }
        }
        const nex = tail.next;
        [head, tail] = myReverse(head, tail); // head是交换后的队尾，tail是交换后队头
        pre.next = head
        tail.next = nex
        pre = tail
        head = tail.next
    }
    return hair.next
};

function myReverse(head, tail){
    let prev = tail.next
    let p = head
    while(prev!==tail) {
        let temp = p.next
        p.next = prev
        prev = p
        p = temp
    }
    return [tail, head]
}