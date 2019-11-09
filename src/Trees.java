import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

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

}

class TreeNode {

    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}
