import apple.laf.JRSUIUtils.Tree;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.stream.Collectors;
import javafx.util.Pair;

/**
 * Created by Sichi Zhang on 2019/11/9.
 */
public class Trees {


    //**********非递归的分层输出的层次遍历**********
    public static void iterativeLevelOrder_1(TreeNode p) {
        if (p == null) {
            return;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(p);
        while (!queue.isEmpty()) {
            int levelNum = queue.size();
            for (int i = 0; i < levelNum; i++) {
                p = queue.poll();
                if (p.left != null) {
                    queue.offer(p.left);
                }
                if (p.right != null) {
                    queue.offer(p.right);
                }
                System.out.println(p.val);
            }

        }
    }

    static void inOrder(TreeNode root) {
        if (root != null) {
            inOrder(root.left);
            System.out.print(root.val);
            inOrder(root.right);
        }
    }

    //  非递归中根遍历
    public static void iterativeInOrder(TreeNode p) {
        if (p == null) {
            return;
        }
        Deque<TreeNode> stack = new ArrayDeque<>();
        while (stack.size() != 0 || p != null) {
            while (p != null) {
                stack.addLast(p);
                p = p.left;
            }
            p = stack.removeLast();
            System.out.println(p.val);
            p = p.right;
        }
    }

    /**
     * Morris中序遍历二叉树(node每次往右移之前打印节点)
     */
    public static void morrisIn(TreeNode head) {
        if (head == null) {
            return;
        }
        TreeNode cur = head;
        TreeNode prev = null;
        while (cur != null) {
            //没有左节点 直接去右节点
            if (cur.left == null) {
                System.out.println(cur.val);
                cur = cur.right;
            } else {
                //找先驱节点，即为中序遍历之前的最后一个节点
                prev = cur.left;
                while (prev.right != null && prev.right != cur) {
                    prev = prev.right;
                }
                if (prev.right == null) {
                    //将中根下面的最后一个节点的右节点设为 predecessor节点 即为返回节点
                    prev.right = cur;
                    cur = cur.left;
                } else {
                    //第二次dfs 发现这是之前设置过的节点，表示已经遍历过左子树，此时复原 并继续遍历右子树
                    prev.right = null;
                    System.out.println(cur.val);
                    cur = cur.right;
                }
            }
        }
    }

    //层次遍历
    public List<List<Integer>> levelOrder(TreeNode root) {

        List<List<Integer>> levels = new ArrayList<List<Integer>>();
        if (root == null) {
            return levels;
        }

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        int level = 0;
        while (!queue.isEmpty()) {
            // start the current level
            levels.add(new ArrayList<>());

            // number of elements in the current level
            int level_length = queue.size();
            for (int i = 0; i < level_length; ++i) {
                TreeNode node = queue.remove();

                // fulfill the current level
                levels.get(level).add(node.val);

                // add child nodes of the current level
                // in the queue for the next level
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            // go to next level
            level++;
        }
        return levels;
    }


    /*
        BST 剪枝到范围之内
     */

    public TreeNode pruningTree(TreeNode root, int min, int max) {
        if (min > max) {
            return null;
        }
        TreeNode newRoot = findRoot(root, min, max);
        pruningLeft(newRoot, min);
        pruningRight(newRoot, max);
        return newRoot;
    }

    //找到剪枝后的新树根
    private TreeNode findRoot(TreeNode root, int min, int max) {
        //新root需要满足min 和max 落在root的两边
        if (root == null || (min <= root.val && max >= root.val)) {
            return root;
        } else if (min < root.val && max < root.val) {
            return findRoot(root.left, min, max);
        } else if (min > root.val && max > root.val) {
            return findRoot(root.right, min, max);
        }
        return null;
    }

    //对最小值做剪枝
    private void pruningLeft(TreeNode root, int min) {
        if (root != null && root.left != null) {
            //如果左孩子不满足要求 需要找到新的左节点
            if (root.left.val < min) {
                TreeNode left = root.left;
                //从当前左孩子的一系列右孩子里，找到第一个大于min的节点作为新的左孩子
                //否则则将左孩子设为null
                while (left != null && left.val < min) {
                    left = left.right;
                }
                root.left = left;
            }
            //对新找到的左孩子递归进行剪枝判断
            pruningLeft(root.left, min);
        }
    }

    //对最大值做剪枝
    private void pruningRight(TreeNode root, int max) {
        //同上逻辑
        if (root != null && root.right != null) {
            if (root.right.val > max) {
                TreeNode right = root.right;
                while (right != null && right.val > max) {
                    right = right.left;
                }
                root.right = right;
            }
            pruningRight(root.right, max);
        }
    }


    /*
        450 BST delete node
        Time H
        Space H
     */
    public int successor(TreeNode root) {
        root = root.right;
        while (root.left != null) {
            root = root.left;
        }
        return root.val;
    }

    /*
    One step left and then always right
    */
    public int predecessor(TreeNode root) {
        root = root.left;
        while (root.right != null) {
            root = root.right;
        }
        return root.val;
    }

    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }

        // delete from the right subtree
        if (key > root.val) {
            root.right = deleteNode(root.right, key);
        }
        // delete from the left subtree
        else if (key < root.val) {
            root.left = deleteNode(root.left, key);
        }
        // delete the current node
        else {
            // the node is a leaf
            if (root.left == null && root.right == null) {
                root = null;
            }
            // the node is not a leaf and has a right child
            else if (root.right != null) {
                root.val = successor(root);
                root.right = deleteNode(root.right, root.val);
            }
            // the node is not a leaf, has no right child, and has a left child
            else {
                root.val = predecessor(root);
                root.left = deleteNode(root.left, root.val);
            }
        }
        return root;
    }

    private TreeNode[] findNodeAndItsParent(TreeNode parent, TreeNode cur, int key) {
        if (cur.val == key) {
            return new TreeNode[]{parent, cur};
        } else if (cur.val < key) {
            return findNodeAndItsParent(cur, cur.right, key);
        } else {
            return findNodeAndItsParent(cur, cur.left, key);
        }
    }


    /*
    lc 236. Lowest Common Ancestor of a Binary Tree
    time O(N)
    space O(H)
     */


    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // Traverse the tree
        this.recurseTree(root, p, q);
        return ans;
    }

    private TreeNode ans;

    private boolean recurseTree(TreeNode currentNode, TreeNode p, TreeNode q) {

        // If reached the end of a branch, return false.
        if (currentNode == null) {
            return false;
        }

        // Left Recursion. If left recursion returns true, map left = 1 else 0
        boolean left = this.recurseTree(currentNode.left, p, q);

        // Right Recursion
        boolean right = this.recurseTree(currentNode.right, p, q);

        // If the current node is one of p or q
        boolean mid = (currentNode == p || currentNode == q);

        // If any two of the flags left, right or mid become True
        if (mid) {
            if (right || left) {
                ans = currentNode;
            }
        } else if (left && right) {
            ans = currentNode;
        }

        // Return true if any one of the three bool values is True.
        return mid || left || right;
    }

    public TreeNode lowestCommonAncestorIterative(TreeNode root, TreeNode p, TreeNode q) {

        // Stack for tree traversal
        Deque<TreeNode> stack = new ArrayDeque<>();

        // HashMap for parent pointers
        Map<TreeNode, TreeNode> parent = new HashMap<>();

        parent.put(root, null);
        stack.push(root);

        // Iterate until we find both the nodes p and q
        while (!parent.containsKey(p) || !parent.containsKey(q)) {

            TreeNode node = stack.pop();

            // While traversing the tree, keep saving the parent pointers.
            if (node.left != null) {
                parent.put(node.left, node);
                stack.push(node.left);
            }
            if (node.right != null) {
                parent.put(node.right, node);
                stack.push(node.right);
            }
        }

        // Ancestors map() for node p.
        Set<TreeNode> ancestors = new HashSet<>();

        // Process all ancestors for node p using parent pointers.
        while (p != null) {
            ancestors.add(p);
            p = parent.get(p);
        }

        // The first ancestor of q which appears in
        // p's ancestor map() is their lowest common ancestor.
        while (!ancestors.contains(q)) {
            q = parent.get(q);
        }
        return q;
    }

    /*
        270. Closest Binary Search Tree Value
        Time O(H)
        Space O(H)
     */
    private Integer closestRes = null;

    public int closestValue(TreeNode root, double target) {
        findCloestValue(root, target);
        return closestRes;
    }

    private void findCloestValue(TreeNode root, double target) {
        if (root != null) {
            if (closestRes == null || Math.abs(root.val - target) < Math.abs(closestRes - target)) {
                closestRes = root.val;
            }
            if (root.val > target) {
                findCloestValue(root.left, target);
            } else if (root.val < target) {
                findCloestValue(root.right, target);
            }
        }
    }

    /*
        222. Count Complete Tree Nodes
        Time d^2 logn*logn  (d 树的深度)
        Space 1
        1.先计算出树的深度
        除了叶子层以外的节点为2^(d-1)个
        最后一层可能有2^d个节点，用二分搜索来找到到底最后一个存在的节点的序号
        1.查找树的深度需要O(d)
        2.查找叶子节点个数总共需要O(log(2^d)*d) = O(d^2);
        总复杂度为d^2 = (logN)^2

     */
    // Return tree depth in O(d) time.
    public int computeDepth(TreeNode node) {
        int d = 0;
        while (node.left != null) {
            node = node.left;
            ++d;
        }
        return d;
    }

    // Last level nodes are enumerated from 0 to 2**d - 1 (left -> right).
    // Return True if last level node idx exists.
    // Binary search with O(d) complexity.
    public boolean exists(int idx, int d, TreeNode node) {
        int left = 0, right = (int) Math.pow(2, d) - 1;
        int pivot;
        for (int i = 0; i < d; ++i) {
            pivot = left + (right - left) / 2;
            if (idx <= pivot) {
                node = node.left;
                right = pivot;
            } else {
                node = node.right;
                left = pivot + 1;
            }
        }
        return node != null;
    }

    public int countNodes(TreeNode root) {
        // if the tree is empty
        if (root == null) {
            return 0;
        }

        int d = computeDepth(root);
        // if the tree contains 1 node
        if (d == 0) {
            return 1;
        }

        // Last level nodes are enumerated from 0 to 2**d - 1 (left -> right).
        // Perform binary search to check how many nodes exist.
        int left = 1, right = (int) Math.pow(2, d) - 1;
        int pivot;
        while (left <= right) {
            pivot = left + (right - left) / 2;
            if (exists(pivot, d, root)) {
                left = pivot + 1;
            } else {
                right = pivot - 1;
            }
        }

        // The tree contains 2**d - 1 nodes on the first (d - 1) levels
        // and left nodes on the last level.
        return (int) Math.pow(2, d) - 1 + left;
    }

    /*
        958. Check Completeness of a Binary Tree
        Time N
        Space N
     */

    public boolean isCompleteTree(TreeNode root) {
        boolean end = false;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode cur = queue.poll();
            if (cur == null) {
                end = true;
            } else {
                if (end) {
                    return false;
                }
                queue.add(cur.left);
                queue.add(cur.right);
            }
        }
        return true;
    }

    /*
        366. Find Leaves of Binary Tree
        Time N
        Space N
     */
    List<List<Integer>> leavesRes = new ArrayList<>();

    public List<List<Integer>> findLeaves(TreeNode root) {
        leavesRes = new ArrayList<>();
        if (root != null) {
            dfs(root);
        }
        return leavesRes;
    }

    private int dfs(TreeNode node) {
        int level = 0;
        if (node.left != null) {
            level = Math.max(dfs(node.left) + 1, level);
        }
        if (node.right != null) {
            level = Math.max(dfs(node.right) + 1, level);
        }
        addToRes(node.val, level);
        return level;
    }

    private void addToRes(int val, int level) {
        if (level >= leavesRes.size()) {
            leavesRes.add(new ArrayList<>());
        }
        leavesRes.get(level).add(val);
    }


    //lc 684
    //time inverse Ackermann function nearly to O(N)
    public int[] findRedundantConnection(int[][] edges) {

        // initialize n isolated islands
        int[] nums = new int[edges.length * 2];
        Arrays.fill(nums, -1);
        int[] res = new int[2];
        // perform union find
        for (int i = 0; i < edges.length; i++) {
            int x = find(nums, edges[i][0]);
            int y = find(nums, edges[i][1]);

            // if two vertices happen to be in the same map
            // then there's a cycle
            //find the last cycle
            if (x == y) {
                res[0] = edges[i][0];
                res[1] = edges[i][1];
            } else {
                // union
                nums[y] = x;

            }
        }
        return res;
    }

    /*
        114. Flatten Binary Tree to Linked List
        Time N
        Space H
     */

    public void flatten(TreeNode root) {
        flattenTree(root);
    }

    private TreeNode flattenTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        if (root.right == null && root.left == null) {
            return root;
        }

        TreeNode leftTail = flattenTree(root.left);
        TreeNode right = root.right;
        if (leftTail != null) {
            leftTail.right = root.right;
            root.right = root.left;
            root.left = null;
        }
        TreeNode rightTail = flattenTree(right);
        return right == null ? leftTail : rightTail;
    }


    /*
    261 Graph valid tree
    dfs solution time O(N) space O(N)
     */
    Map<Integer, List<Integer>> connectMap;
    Set<Integer> isVisited;

    public boolean validTree(int n, int[][] edges) {

        isVisited = new HashSet<>();
        connectMap = new HashMap<>();
        //get a adjacent list
        for (int[] edge : edges) {
            List<Integer> connection1 = connectMap.getOrDefault(edge[0], new ArrayList<>());
            List<Integer> connection2 = connectMap.getOrDefault(edge[1], new ArrayList<>());
            connection1.add(edge[1]);
            connection2.add(edge[0]);
            connectMap.put(edge[0], connection1);
            connectMap.put(edge[1], connection2);
        }
        return n == 1 || dfs(0, -1) && isVisited.size() == n;

    }

    private boolean dfs(int cur, int parent) {
        if (!connectMap.containsKey(cur)) {
            return false;
        }
        isVisited.add(cur);
        for (int neighbour : connectMap.get(cur)) {
            if (neighbour == parent) {
                continue;
            }
            if (isVisited.contains(neighbour)) {
                return false;
            }
            if (!dfs(neighbour, cur)) {
                return false;
            }
        }
        return true;
    }

    //space(N) time(N)
    public boolean validTreeUnionFind(int n, int[][] edges) {
        // initialize n isolated islands
        int[] nums = new int[n];
        Arrays.fill(nums, -1);

        // perform union find
        for (int i = 0; i < edges.length; i++) {
            int x = find(nums, edges[i][0]);
            int y = find(nums, edges[i][1]);

            // if two vertices happen to be in the same map
            // then there's a cycle
            if (x == y) {
                return false;
            }

            // union
            nums[y] = x;
        }

        return edges.length == n - 1;
    }

    int find(int nums[], int i) {
        if (nums[i] == -1) {
            return i;
        }
        return find(nums, nums[i]);
    }

    //lc617 合并两棵树
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return null;
        } else if (t1 != null && t2 == null) {
            return t1;
        } else if (t1 == null && t2 != null) {
            return t2;
        } else {
            TreeNode node = new TreeNode(t1.val + t2.val);
            node.left = mergeTrees(t1.left, t2.left);
            node.right = mergeTrees(t1.right, t2.right);
            return node;
        }
    }

    public TreeNode mergeTreesIterative(TreeNode t1, TreeNode t2) {
        if (t1 == null) {
            return t2;
        }
        Stack<TreeNode[]> stack = new Stack<>();
        stack.push(new TreeNode[]{t1, t2});
        while (!stack.isEmpty()) {
            TreeNode[] t = stack.pop();
            if (t[0] == null || t[1] == null) {
                continue;
            }
            t[0].val += t[1].val;
            if (t[0].left == null) {
                t[0].left = t[1].left;
            } else {
                stack.push(new TreeNode[]{t[0].left, t[1].left});
            }
            if (t[0].right == null) {
                t[0].right = t[1].right;
            } else {
                stack.push(new TreeNode[]{t[0].right, t[1].right});
            }
        }
        return t1;
    }


    /*
        124 Binary Tree Maximum Path Sum 任意path sum最大，不一定要经过root
        Time O(N) a full traverse
        Space O(H)
     */
    private class MaxPathSum {

        int res = Integer.MIN_VALUE;

        public int maxPathSum(TreeNode root) {
            maxSumForOneNode(root);
            return res;
        }

        //the max sum that pass this node and its not a root
        private int maxSumForOneNode(TreeNode node) {
            if (node == null) {
                return 0;
            }
            int left = Math.max(maxSumForOneNode(node.left), 0);
            int right = Math.max(maxSumForOneNode(node.right), 0);
            int maxChild = Math.max(left, right);
            //if this node is the top one.
            res = Math.max(res, node.val + left + right);
            return node.val + maxChild;
        }
    }


    public void addLeaves(List<Integer> res, TreeNode root) {
        if (isLeaf(root)) {
            res.add(root.val);
        } else {
            if (root.left != null) {
                addLeaves(res, root.left);
            }
            if (root.right != null) {
                addLeaves(res, root.right);
            }
        }
    }

    /*
        545. Boundary of Binary Tree
        time O(n) space O(N) stack
     */
    public List<Integer> boundaryOfBinaryTree(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        if (!isLeaf(root)) {
            res.add(root.val);
        }
        TreeNode t = root.left;
        //add left boundary
        while (t != null) {
            if (!isLeaf(t)) {
                res.add(t.val);
            }
            if (t.left != null) {
                t = t.left;
            } else {
                t = t.right;
            }
        }
        addLeaves(res, root);
        // //add right boundary
        Stack<Integer> s = new Stack<>();
        t = root.right;
        while (t != null) {
            if (!isLeaf(t)) {
                s.push(t.val);
            }
            if (t.right != null) {
                t = t.right;
            } else {
                t = t.left;
            }
        }
        while (!s.empty()) {
            res.add(s.pop());
        }
        return res;
    }

    //垂直方向 从左到右 从上到下遍历


    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        //or use treemap to keep order
        Map<Integer, List<Integer>> map = new HashMap<>();
        int min = 0;
        int max = 0;
        if (root == null) {
            return res;
        }
        Pair<TreeNode, Integer> pR = new Pair<>(root, 0);
        Queue<Pair<TreeNode, Integer>> queue = new LinkedList<>();
        queue.add(pR);
        while (!queue.isEmpty()) {
            // start the current level
            Pair<TreeNode, Integer> cur = queue.remove();
            min = Math.min(cur.getValue(), min);
            max = Math.max(cur.getValue(), max);
            List<Integer> curList = map.getOrDefault(cur.getValue(), new ArrayList<>());
            curList.add(cur.getKey().val);
            map.put(cur.getValue(), curList);
            if (cur.getKey().left != null) {
                queue.add(new Pair<>(cur.getKey().left, cur.getValue() - 1));
            }
            if (cur.getKey().right != null) {
                queue.add(new Pair<>(cur.getKey().right, cur.getValue() + 1));
            }
        }
        for (int i = min; i <= max; i++) {
            res.add(map.get(i));
        }
        return res;


    }


    private class Node {

        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    /*
        104. Maximum Depth of Binary Tree
        Time N
        Space H
     */
    public int maxDepth(TreeNode root) {
        return dfs(root, 0);
    }

    int dfs(TreeNode t, int dep) {
        if (t != null) {
            dep++;
            return Math.max(dfs(t.left, dep), dfs(t.right, dep));
        } else {
            return dep;
        }
    }

    /*
        979. Distribute Coins in Binary Tree
        time N
        space H
     */
    int steps;

    public int distributeCoins(TreeNode root) {
        steps = 0;
        dfsdistributeCoins(root);
        return steps;
    }

    public int dfsdistributeCoins(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int L = dfs(node.left);
        int R = dfs(node.right);
        steps += Math.abs(L) + Math.abs(R);
        return node.val + L + R - 1;
    }

    /*
        105. Construct Binary Tree from Preorder and Inorder Traversal
        Time N
        Space N
     */
    class BuildTree {

        int preCur;
        int[] preorder;
        Map<Integer, Integer> indexMap;

        public TreeNode buildTree(int[] preorder, int[] inorder) {
            if (preorder.length == 0 || preorder.length != inorder.length) {
                return null;
            }
            this.preorder = preorder;
            preCur = 0;
            indexMap = new HashMap<>();
            for (int i = 0; i < inorder.length; i++) {
                indexMap.put(inorder[i], i);
            }
            return dfs(0, inorder.length - 1);
        }

        public TreeNode dfs(int inLeft, int inRight) {
            if (inLeft > inRight) {
                return null;
            }
            TreeNode root = new TreeNode(preorder[preCur++]);
            int rootIndex = indexMap.get(root.val);
            root.left = dfs(inLeft, rootIndex - 1);
            root.right = dfs(rootIndex + 1, inRight);
            return root;
        }
    }


    //lc 117 next right point tree is not perfect
    //straightforward 为每个节点寻找下一个节点
    public Node connect2(Node root) {
        Node head = root;
        Node level_start = root;
        while (level_start != null) {
            Node cur = level_start;
            while (cur != null) {
                if (cur.left != null) {
                    if (cur.right != null) {
                        cur.left.next = cur.right;
                    } else {
                        //find next right
                        Node nextRight = cur.next;
                        while (nextRight != null) {
                            if (nextRight.left == null && nextRight.right == null) {
                                nextRight = nextRight.next;
                            } else {
                                break;
                            }
                        }
                        cur.left.next = nextRight == null ? null
                            : nextRight.left != null ? nextRight.left : nextRight.right;
                    }
                }

                if (cur.right != null) {
                    Node nextRight = cur.next;
                    while (nextRight != null) {
                        if (nextRight.left == null && nextRight.right == null) {
                            nextRight = nextRight.next;
                        } else {
                            break;
                        }
                    }
                    cur.right.next = nextRight == null ? null
                        : nextRight.left != null ? nextRight.left : nextRight.right;
                }
                cur = cur.next;
            }
            while (level_start.next != null) {
                if (level_start.left == null && level_start.right == null) {
                    level_start = level_start.next;
                } else {
                    break;
                }
            }

            level_start = level_start.left == null && level_start.right == null ? null
                : level_start.left == null ? level_start.right : level_start.left;
        }
        return head;
    }

    //使用dummyhead节点 逐个链接同层次的每个节点的左右节点寻找next
    public Node connect2DummyHead(Node root) {
        Node curP = root;
        Node nextDummyHead = new Node();
        Node p = nextDummyHead;
        while (curP != null) {
            if (curP.left != null) {
                p.next = curP.left;
                p = p.next;
            }
            if (curP.right != null) {
                p.next = curP.right;
                p = p.next;
            }
            if (curP.next != null) {
                curP = curP.next;
            } else {
                curP = nextDummyHead.next;
                nextDummyHead.next = null;
                p = nextDummyHead;
            }
        }
        return root;
    }


    //lc 116 next right point tree is perfect
    public Node connect1(Node root) {
        Node head = root;
        Node level_start = root;
        while (level_start != null) {
            Node cur = level_start;
            while (cur != null) {
                if (cur.left != null) {
                    cur.left.next = cur.right;
                }
                if (cur.right != null && cur.next != null) {
                    cur.right.next = cur.next.left;
                }

                cur = cur.next;
            }
            level_start = level_start.left;
        }
        return head;
    }


    /*
        113. Path Sum II
        from root to leaf
        Time O(N^2) N for traversal another N for array copy
        space n or logn
     */
    private List<List<Integer>> res;

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) {
            return new ArrayList<List<Integer>>();
        }
        res = new ArrayList<>();
        dfs(root, sum, new ArrayList<Integer>());
        return res;

    }

    private void dfs(TreeNode node, int sum, List<Integer> path) {
        if (node == null) {
            return;
        }
        path.add(node.val);
        sum -= node.val;
        if (isLeaf(node) && sum == 0) {
            res.add(new ArrayList<>(path));
        } else {
            dfs(node.left, sum, path);
            dfs(node.right, sum, path);
        }
        path.remove(path.size() - 1);

    }


    private void findPathSumDFS(List<Integer> path, TreeNode node, int remainder) {
        if (node == null) {
            return;
        }
        path.add(node.val);
        remainder -= node.val;
        if (isLeaf(node) && remainder == 0) {
            res.add(path);
            return;
        }
        findPathSumDFS(new ArrayList<>(path), node.left, remainder);
        findPathSumDFS(new ArrayList<>(path), node.right, remainder);
    }

    //from up to down
    private int pathSumCount = 0;

    /*
        437. Path Sum III use path or Optimize Hash map solution
        Time O(NlogN) path length should be logn
     */
    public int pathSum3(TreeNode root, int sum) {
        List<Integer> path = new ArrayList<>();
        findPathSum3DFS(path, root, sum);
        return pathSumCount;
    }

    private void findPathSum3DFS(List<Integer> path, TreeNode node, int sum) {
        if (node == null) {
            return;
        }
        path.add(node.val);
        int total = 0;
        for (int i = path.size() - 1; i >= 0; i--) {
            total += path.get(i);
            if (total == sum) {
                pathSumCount++;
            }
        }
        findPathSumDFS(new ArrayList<>(path), node.left, sum);
        findPathSumDFS(new ArrayList<>(path), node.right, sum);
    }


    /*
        Time O(N) dfs all nodes
        Space O(N) for map and recursion stack
     */
    public int pathSum3Opt(TreeNode root, int sum) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);  //Default sum = 0 has one count
        return backtrack(root, 0, sum, map);
    }

    //BackTrack one pass
    public int backtrack(TreeNode root, int sum, int target, Map<Integer, Integer> map) {
        if (root == null) {
            return 0;
        }
        sum += root.val;
        int res = map
            .getOrDefault(sum - target, 0);    //See if there is a subarray sum equals to target
        map.put(sum, map.getOrDefault(sum, 0) + 1);
        //Extend to left and right child
        res += backtrack(root.left, sum, target, map) + backtrack(root.right, sum, target, map);
        map.put(sum, map.get(sum) - 1);   //Remove the current node so it wont affect other path
        return res;
    }


    /*
        103. Binary Tree Zigzag Level Order Traversal
        Time N
        Space  BFS N DFS H
     */
    public List<List<Integer>> zigzagLevelOrderBFS(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> bfsHelper = new LinkedList<>();
        if (root != null) {
            bfsHelper.add(root);
        }
        while (!bfsHelper.isEmpty()) {
            int size = bfsHelper.size();
            LinkedList<Integer> levelList = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = bfsHelper.poll();
                if (res.size() % 2 == 0) {
                    levelList.addLast(cur.val);
                } else {
                    levelList.addFirst(cur.val);
                }
                if (cur.left != null) {
                    bfsHelper.offer(cur.left);
                }
                if (cur.right != null) {
                    bfsHelper.offer(cur.right);
                }
            }
            res.add(levelList);
        }
        return res;
    }


    protected void DFS(TreeNode node, int level, List<List<Integer>> results) {
        if (level >= results.size()) {
            LinkedList<Integer> newLevel = new LinkedList<Integer>();
            newLevel.add(node.val);
            results.add(newLevel);
        } else {
            if (level % 2 == 0) {
                results.get(level).add(node.val);
            } else {
                results.get(level).add(0, node.val);
            }
        }

        if (node.left != null) {
            DFS(node.left, level + 1, results);
        }
        if (node.right != null) {
            DFS(node.right, level + 1, results);
        }
    }

    public List<List<Integer>> zigzagLevelOrderDFS(TreeNode root) {
        if (root == null) {
            return new ArrayList<List<Integer>>();
        }
        List<List<Integer>> results = new ArrayList<List<Integer>>();
        DFS(root, 0, results);
        return results;
    }

    /*
        101. Symmetric Tree
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isMirror(root.left, root.right);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        return (t1.val == t2.val)
            && isMirror(t1.right, t2.left)
            && isMirror(t1.left, t2.right);
    }

    /*
        98. Validate Binary Search Tree
        time n
        space n
     */
    public boolean isValidBSTInOrder(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        Integer prev = null;
        TreeNode cur = root;
        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            if (prev != null && cur.val <= prev) {
                return false;
            }
            prev = cur.val;
            cur = cur.right;
        }
        return true;
    }

    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        } else {

            return checkBSTHelper(Integer.MIN_VALUE, Integer.MAX_VALUE, root, true, true);
        }
    }

    boolean checkBSTHelper(int min, int max, TreeNode root, boolean leftMost, boolean rightMost) {
        if (root == null) {
            return true;
        }
        if (root.left != null && (root.left.val >= root.val || (root.left.val <= min
            && !leftMost))) {
            return false;
        }
        if (root.right != null && (root.right.val <= root.val || (root.right.val >= max
            && !rightMost))) {
            return false;
        }
        return checkBSTHelper(min, root.val, root.left, leftMost, false) && checkBSTHelper(root.val,
            max, root.right, false, rightMost);
    }

    /*
        111. Minimum Depth of Binary Tree
        Time N
        Space N
     */
    public int minDepthBFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> bfsHelper = new LinkedList<>();
        bfsHelper.add(root);
        int depth = 0;
        while (!bfsHelper.isEmpty()) {
            int size = bfsHelper.size();
            depth++;
            for (int i = 0; i < size; i++) {
                TreeNode n = bfsHelper.poll();
                if (n.left != null) {
                    bfsHelper.offer(n.left);
                }
                if (n.right != null) {
                    bfsHelper.offer(n.right);
                }
                if (n.left == null && n.right == null) {
                    return depth;
                }
            }
        }
        return depth;
    }

    public int minDepthDFS(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if ((root.left == null) && (root.right == null)) {
            return 1;
        }

        int min_depth = Integer.MAX_VALUE;
        if (root.left != null) {
            min_depth = Math.min(minDepthDFS(root.left), min_depth);
        }
        if (root.right != null) {
            min_depth = Math.min(minDepthDFS(root.right), min_depth);
        }

        return min_depth + 1;

    }

    /*
        404. Sum of Left Leaves
        Time O(N)
        space O(N) or logn
     */
    public int sumOfLeftLeaves(TreeNode root) {
        int sum = 0;
        if (root == null) {
            return 0;
        }
        if (root.left != null && isLeaf(root.left)) {
            sum += root.left.val;
        } else {
            sum += sumOfLeftLeaves(root.left);
        }
        sum += sumOfLeftLeaves(root.right);
        return sum;
    }

    public boolean isLeaf(TreeNode n) {
        return n.left == null && n.right == null;
    }


    //模拟二叉搜索树的中根遍历来构造BST
    /*
        109. Convert Sorted List to Binary Search Tree
        time N
        space logN
     */
    private class SortedListToBST {

        private ListNode head;

        private int findSize(ListNode head) {
            ListNode ptr = head;
            int c = 0;
            while (ptr != null) {
                ptr = ptr.next;
                c += 1;
            }
            return c;
        }

        private TreeNode convertListToBST(int l, int r) {
            // Invalid case
            if (l > r) {
                return null;
            }

            int mid = (l + r) / 2;

            // First step of simulated inorder traversal. Recursively form
            // the left half
            TreeNode left = this.convertListToBST(l, mid - 1);

            // Once left half is traversed, process the current node
            TreeNode node = new TreeNode(this.head.val);
            node.left = left;

            // Maintain the invariance mentioned in the algorithm
            this.head = this.head.next;

            // Recurse on the right hand side and form BST out of them
            node.right = this.convertListToBST(mid + 1, r);
            return node;
        }

        public TreeNode sortedListToBST(ListNode head) {
            // Get the size of the linked list first
            int size = this.findSize(head);

            this.head = head;

            // Form the BST now that we know the size
            return convertListToBST(0, size - 1);
        }
    }


    /*
        108. Convert Sorted Array to Binary Search Tree
        time N
        Space N for output  logN for recursion stack
     */
    int[] nums;

    public TreeNode sortedArrayToBST(int[] nums) {
        this.nums = nums;
        return buildTree(0, nums.length - 1);
    }

    TreeNode buildTree(int l, int r) {
        if (l > r) {
            return null;
        }
        int mid = (l + r) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = buildTree(l, mid - 1);
        root.right = buildTree(mid + 1, r);
        return root;
    }

    /*
    lc 543  Diameter of Binary Tree
    dfs the tree and calculate each depth. return the max depth sum from left child and right child.
    Time O(N) whole tree dfs traversal
    Space O(LogN) or O(H) stack need to store maximum the number of height of nodes.
     */
    public int diameterOfBinaryTree(TreeNode root) {
        int[] ans = new int[]{1};
        depthOfTree(root, ans);
        return ans[0] - 1;
    }

    public int depthOfTree(TreeNode node, int[] ans) {
        if (node == null) {
            return 0;
        }
        int L = depthOfTree(node.left, ans);
        int R = depthOfTree(node.right, ans);
        ans[0] = Math.max(ans[0], L + R + 1);
        return Math.max(L, R) + 1;
    }

    /*
    树右边看
    lc 199. Binary Tree Right Side View
    dfs遍历 根 右 左顺序，并记录当前深度，使用set记录到过的层数，如果是新到一层则加入到res
    bfs同理
    time O(N)
    space O(H) best case H = logN worst H = N
     */
    List<Integer> res199;

    public List<Integer> rightSideView(TreeNode root) {
        res199 = new ArrayList<>();
        dfsRightSide(root, 1);
        return res199;
    }

    private void dfsRightSide(TreeNode node, int level) {
        if (node == null) {
            return;
        }
        if (level > res199.size()) {
            res199.add(node.val);
        }
        dfsRightSide(node.right, level + 1);
        dfsRightSide(node.left, level + 1);
    }

    /*
        112. Path Sum
        Time O(N)
        Space O(N) best O(logN)
        存在一个path sum等于target path sum就是从root到leaf
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return root.val == sum;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    //8 10 14 3 6 7 70 30 13 1 4
    //树对角线方向遍历
    public static void diagonalPrint(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {
            TreeNode cur = q.peek();
            while (cur != null) {
                System.out.print(cur.val + " ");
                if (cur.left != null) {
                    q.add(cur.left);
                }
                cur = cur.right;
            }
            q.poll();
        }
    }


    public void inorder(TreeNode root, List<Integer> nums) {
        if (root == null) {
            return;
        }
        inorder(root.left, nums);
        nums.add(root.val);
        inorder(root.right, nums);
    }

    public int[] findTwoSwapped(List<Integer> nums) {
        int n = nums.size();
        int x = -1, y = -1;
        for (int i = 0; i < n - 1; ++i) {
            if (nums.get(i + 1) < nums.get(i)) {
                //假如颠倒的两个数挨着，就刚好是第一次取得x，y 不然y是下一个比前面一个数大的数
                y = nums.get(i + 1);
                // first swap occurence
                if (x == -1) {
                    //x为第一个确定的数
                    x = nums.get(i);
                }
                // second swap occurence
                else {
                    break;
                }
            }
        }
        return new int[]{x, y};
    }

    public void recover(TreeNode r, int count, int x, int y) {
        if (r != null) {
            if (r.val == x || r.val == y) {
                r.val = r.val == x ? y : x;
                if (--count == 0) {
                    return;
                }
            }
            recover(r.left, count, x, y);
            recover(r.right, count, x, y);
        }
    }

    //Time&Space:O(N)  用morris traveral 可以变成O(1)
    public void recoverTree(TreeNode root) {
        List<Integer> nums = new ArrayList();
        inorder(root, nums);
        int[] swapped = findTwoSwapped(nums);
        recover(root, 2, swapped[0], swapped[1]);
    }

    public void swap(TreeNode a, TreeNode b) {
        int tmp = a.val;
        a.val = b.val;
        b.val = tmp;
    }

    //use prev to record the last node, instead of store the inorder array.
    //Time O(N) finish a dfs  Space O(H) at most stored H node in stack.
    public void recoverTreeOneDfs(TreeNode root) {
        Deque<TreeNode> stack = new ArrayDeque();
        TreeNode x = null, y = null, pred = null;

        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.add(root);
                root = root.left;
            }
            root = stack.removeLast();
            if (pred != null && root.val < pred.val) {
                y = root;
                if (x == null) {
                    x = pred;
                } else {
                    break;
                }
            }
            pred = root;
            root = root.right;
        }

        swap(x, y);
    }

}

class Codec {

    /*
    Time complexity : in both serialization and deserialization functions,
    we visit each node exactly once, thus the time complexity is O(N)O(N),
     where NN is the number of nodes, i.e. the size of tree.
     */
    class CodecBiT {

        public StringBuilder rserialize(TreeNode root, StringBuilder str) {
            // Recursive serialization.
            if (root == null) {
                str.append("#,");
            } else {
                str.append(root.val);
                str.append(",");
                rserialize(root.left, str);
                rserialize(root.right, str);
            }
            return str;
        }

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            return rserialize(root, new StringBuilder()).toString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            String[] data_array = data.split(",");
            List<String> data_list = new LinkedList<String>(Arrays.asList(data_array));
            return rdeserialize(data_list);
        }

        public TreeNode rdeserialize(List<String> l) {
            // Recursive deserialization.
            if (l.get(0).equals("#")) {
                l.remove(0);
                return null;
            }

            TreeNode root = new TreeNode(Integer.valueOf(l.get(0)));
            l.remove(0);
            root.left = rdeserialize(l);
            root.right = rdeserialize(l);

            return root;
        }
    }

    class CodecNTree {

        class Node {

            public int val;
            public List<Node> children;

            public Node() {
            }

            public Node(int _val, List<Node> _children) {
                val = _val;
                children = _children;
            }
        }

        // Encodes a tree to a single string.
        public String serialize(Node root) {
            List<String> list = new LinkedList<>();
            serializeHelper(root, list);
            return String.join(",", list);
        }

        private void serializeHelper(Node root, List<String> list) {
            if (root == null) {
                return;
            } else {
                list.add(String.valueOf(root.val));
                list.add(String.valueOf(root.children.size()));
                for (Node child : root.children) {
                    serializeHelper(child, list);
                }
            }
        }

        // Decodes your encoded data to tree.
        public Node deserialize(String data) {
            if (data.isEmpty()) {
                return null;
            }

            String[] ss = data.split(",");
            List<String> q = new LinkedList<>(Arrays.asList(ss));
            return deserializeHelper(q);
        }

        private Node deserializeHelper(List<String> q) {
            Node root = new Node();
            root.val = Integer.parseInt(q.remove(0));
            int size = Integer.parseInt(q.remove(0));
            root.children = new ArrayList<Node>(size);
            for (int i = 0; i < size; i++) {
                root.children.add(deserializeHelper(q));
            }
            return root;
        }
    }


}


class TreeNode {

    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}

class BSTIterator {

    Stack<TreeNode> stack;

    public BSTIterator(TreeNode root) {

        // Stack for the recursion simulation
        this.stack = new Stack<TreeNode>();

        // Remember that the algorithm starts with a call to the helper function
        // with the root node as the input
        this._leftmostInorder(root);
    }

    private void _leftmostInorder(TreeNode root) {

        // For a given node, add all the elements in the leftmost branch of the tree
        // under it to the stack.
        while (root != null) {
            this.stack.push(root);
            root = root.left;
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        // Tile at the top of the stack is the next smallest element
        TreeNode topmostNode = this.stack.pop();

        // Need to maintain the invariant. If the node has a right child, call the
        // helper function for the right child
        if (topmostNode.right != null) {
            this._leftmostInorder(topmostNode.right);
        }

        return topmostNode.val;
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return this.stack.size() > 0;
    }
}

