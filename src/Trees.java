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
                    prev.right = cur;
                    cur = cur.left;
                } else {
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
            levels.add(new ArrayList<Integer>());

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

    //time O(n) space O(N)
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


    private List<List<Integer>> res = new ArrayList<>();

    //from root to leaf
    public List<List<Integer>> pathSum2(TreeNode root, int sum) {
        List<Integer> path = new ArrayList<>();
        findPathSumDFS(path, root, sum);
        return res;
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

    //O(N)
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


    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        Deque<Pair<TreeNode, Integer>> q = new ArrayDeque<>();
        Pair<TreeNode, Integer> p = new Pair<>(root, 0);
        q.addLast(p);
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        do {
            Pair<TreeNode, Integer> cur = q.removeFirst();
            List<Integer> res1 = new ArrayList<>();
            while (cur.getValue() % 2 == 0) {
                res1.add(cur.getKey().val);
                if (cur.getKey().left != null) {
                    q.addLast(new Pair<>(cur.getKey().left, cur.getValue() + 1));
                }
                if (cur.getKey().right != null) {
                    q.addLast(new Pair<>(cur.getKey().right, cur.getValue() + 1));
                }
                if (!q.isEmpty()) {
                    if (q.peekFirst().getValue() % 2 != 0) {
                        break;
                    } else {
                        cur = q.removeFirst();
                    }
                } else {
                    break;
                }
            }
            if (!res1.isEmpty()) {
                res.add(res1);
            }

            if (q.isEmpty()) {
                break;
            } else {
                cur = q.removeLast();
            }

            res1 = new ArrayList<>();
            while (cur.getValue() % 2 == 1) {
                res1.add(cur.getKey().val);
                if (cur.getKey().right != null) {
                    q.addFirst(new Pair<>(cur.getKey().right, cur.getValue() + 1));
                }
                if (cur.getKey().left != null) {
                    q.addFirst(new Pair<>(cur.getKey().left, cur.getValue() + 1));
                }
                if (!q.isEmpty()) {
                    if (q.peekLast().getValue() % 2 != 1) {
                        break;
                    } else {
                        cur = q.removeLast();
                    }
                } else {
                    break;
                }
            }
            if (!res1.isEmpty()) {
                res.add(res1);
            }
        } while (!q.isEmpty());
        return res;
    }

    //O(n)
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


    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        if (nums.length == 1) {
            return new TreeNode(nums[0]);
        }
        int mid = nums.length / 2;
        TreeNode root = new TreeNode(nums[mid]);
        if (mid > 0) {
            root.left = sortedArrayToBST(Arrays.copyOf(nums, mid));
        }
        if (mid < nums.length - 1) {
            root.right = sortedArrayToBST(Arrays.copyOfRange(nums, mid + 1, nums.length));
        }
        return root;
    }

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

    //树右边看
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        if (root == null) {
            return res;
        }
        dfsOfRightSide(root, 0, set, res);
        return res;
    }

    void dfsOfRightSide(TreeNode x, int level, Set<Integer> set, List<Integer> res) {
        if (x == null) {
            return;
        }

        if (!set.contains(level)) {
            set.add(level);
            res.add(x.val);
        }

        dfsOfRightSide(x.right, level + 1, set, res);
        dfsOfRightSide(x.left, level + 1, set, res);
    }

    //存在一个path sum等于target path sum就是从root到leaf
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
                y = nums.get(i + 1);
                // first swap occurence
                if (x == -1) {
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

}

class TreeNode {

    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}
