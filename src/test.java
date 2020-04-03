import java.util.*;
import javafx.util.Pair;

interface a {

}

class Solution {

    int[][] dir = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

    public int numIslands(char[][] grid) {
        int row = grid.length;
        if (row == 0) {
            return 0;
        }
        int col = grid[0].length;
        UnionFind uf = new UnionFind(row * col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1') {
                    for (int t = 0; t < 4; t++) {
                        int newRow = dir[t][0] + i;
                        int newCol = dir[t][1] + j;
                        if (isInGrid(newRow, newCol, grid) && grid[newRow][newCol] == '1') {
                            uf.union(i * col + j, newRow * col + newCol);
                        }
                    }
                }
            }
        }
        int count = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1' && uf.find(i * col + j) == i * col + j) {
                    count++;
                }
            }
        }

        return count;
    }

    private boolean isInGrid(int row, int col, char[][] grid) {
        return row >= 0 && row < grid.length && col >= 0 && col < grid[0].length;
    }
}

public class test {

    int a;


    public static void main(String[] args) {
        double a = 1 + 0.5;
    }

    public static boolean arraySumInAnotherArray(int[] a, int[] b) {
        Set<Integer> setB = new HashSet<>();
        for (int num : b) {
            setB.add(num);
        }
        for (int i = 0; i < a.length - 1; i++) {
            for (int j = i + 1; j < a.length; j++) {
                if (setB.contains(a[i] + a[j])) {
                    return true;
                }
            }
        }
        return false;
    }


    /*
        152. Maximum Product Subarray
        每次储存乘到上一个数的最小和最大值，更新当前数乘上个数或者不乘的当前最小最大值
        更新最大答案
        Time O(N)
        Space O(1)
     */
    public int maxProduct(int[] nums) {
        int prevMax = nums[0], prevMin = nums[0];
        int res = prevMax;
        for (int i = 1; i < nums.length; i++) {
            int tempMax = Math.max(nums[i], Math.max(nums[i] * prevMax, nums[i] * prevMin));
            prevMin = Math.min(nums[i], Math.min(nums[i] * prevMax, nums[i] * prevMin));
            prevMax = tempMax;
            res = Math.max(res, prevMax);
        }
        return res;
    }

    public static int findDateInAYear(int year) {
        Calendar c = new GregorianCalendar();
        c.set(year, Calendar.OCTOBER, 1);
        int diff = c.get(Calendar.DAY_OF_WEEK) - Calendar.TUESDAY;
        if (diff > 0) {
            int offSet = 14 - diff;
            c.set(year, Calendar.OCTOBER, offSet + 1);
        } else {
            c.set(year, Calendar.OCTOBER, 8 - diff);
        }
        System.out.println(c.getTime());
        return c.get(Calendar.DAY_OF_MONTH);
    }

    static class SlidingPuzzleMyOwn {

        public int slidingPuzzle(int[][] board) {
            //check if the board valid
            if (board == null || board.length == 0 || board[0].length == 0) {
                return -1;
            }

            int N = board.length;
            int M = board[0].length;

            //build the final state of this problem.
            int[][] targetBoard = new int[N][M];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    targetBoard[i][j] = i * M + j;
                }
            }
            State targetState = new State(targetBoard, 0, 0);

            int currentSteps = -1;

            Set<State> visitedState = new HashSet<>();
            Queue<State> queue = new LinkedList();

            //build the original state
            State original = null;
            int[][] directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[0].length; j++) {
                    if (board[i][j] == 0) {
                        original = new State(deepCopyOfBoardArray(board), i, j);
                        break;
                    }
                }
            }
            queue.offer(original);
            visitedState.add(original);

            while (!queue.isEmpty()) {
                int size = queue.size();
                //one more level bfs, means one more step.
                currentSteps++;
                for (int i = 0; i < size; i++) {
                    State cur = queue.poll();
                    if (targetState.equals(cur)) {
                        return currentSteps;
                    }

                    for (int j = 0; j < 4; j++) {
                        int[] direction = directions[j];
                        int nextRow = cur.posOf0[0] + direction[0];
                        int nextCol = cur.posOf0[1] + direction[1];
                        if (inBoard(nextRow, nextCol, cur.board)) {

                            int[][] newBoard = deepCopyOfBoardArray(cur.board);
                            //swap cur 0's position with new position.
                            int temp = newBoard[nextRow][nextCol];
                            newBoard[nextRow][nextCol] = newBoard[cur.posOf0[0]][cur.posOf0[1]];
                            newBoard[cur.posOf0[0]][cur.posOf0[1]] = temp;
                            State newState = new State(newBoard, nextRow, nextCol);

                            if (!visitedState.contains(newState)) {
                                queue.offer(newState);
                                visitedState.add(newState);
                            }
                        }
                    }
                }
            }

            return -1;
        }

        private boolean inBoard(int row, int col, int[][] grid) {
            return row >= 0 && col >= 0 && row < grid.length && col < grid[0].length;
        }

        private int[][] deepCopyOfBoardArray(int[][] board) {
            int[][] res = new int[board.length][board[0].length];
            for (int i = 0; i < res.length; i++) {
                for (int j = 0; j < res[0].length; j++) {
                    res[i][j] = board[i][j];
                }
            }
            return res;
        }


        private class State {

            int[][] board;
            int[] posOf0;

            State(int[][] board, int rowOf0, int colOf0) {
                this.board = board;
                this.posOf0 = new int[]{rowOf0, colOf0};
            }

            /*
                 using its String format in hash function
             */
            @Override
            public int hashCode() {
                return toString().hashCode();
            }

            /*
                convert the 2d array to String.
             */
            @Override
            public String toString() {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < board.length; i++) {
                    for (int j = 0; j < board[0].length; j++) {
                        sb.append(board[i][j]);
                        sb.append("|");
                    }
                }
                return sb.toString();
            }

            /*
                Determine the equality by its String format.
             */
            @Override
            public boolean equals(Object obj) {
                return toString().equals(obj.toString());
            }
        }
    }


    static int pow(int x, int n) {
        int sum = 1;
        int tmp = x;
        while (n != 0) {
            if ((n & 1) == 1) {
                sum *= tmp;
            }
            tmp *= tmp;
            n = n >> 1;
        }

        return sum;
    }


    public static int calculate(String s) {
        s = s.replaceAll("\\s+", "");
        Stack<Integer> stack = new Stack<>();
        char sign = '+';
        for (int i = 0; i < s.length(); ) {
            char c = s.charAt(i);
            if (c == '(') {
                // find the block and use the recursive to solve
                int l = 1;
                int j = i + 1;
                while (j < s.length() && l > 0) {
                    if (s.charAt(j) == '(') {
                        l++;
                    } else if (s.charAt(j) == ')') {
                        l--;
                    }
                    j++;
                }
                int blockValue = calculate(s.substring(i + 1, j - 1));
                i = j;
                if (sign == '+') {
                    stack.push(blockValue);
                } else if (sign == '-') {
                    stack.push(-blockValue);
                } else if (sign == '*') {
                    stack.push(stack.pop() * blockValue);
                } else if (sign == '/') {
                    stack.push(stack.pop() / blockValue);
                }
            } else if (Character.isDigit(c)) {
                int j = i;
                int value = 0;
                while (j < s.length() && Character.isDigit(s.charAt(j))) {
                    value = 10 * value + (s.charAt(j) - '0');
                    j++;
                }
                i = j;
                if (sign == '+') {
                    stack.push(value);
                } else if (sign == '-') {
                    stack.push(-value);
                } else if (sign == '*') {
                    stack.push(stack.pop() * value);
                } else if (sign == '/') {
                    stack.push(stack.pop() / value);
                }
            } else {
                sign = c;
                i++;
            }
        }
        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }


    public static void main99(String[] args) {
        Scanner sc = new Scanner(System.in);
        int T = sc.nextInt();
        Map<List<Integer>, Integer> cases = new HashMap<>();
        for (int i = 0; i < T; i++) {
            List<Integer> xi = new ArrayList<>();
            int sum = 0;
            int n = sc.nextInt();
            for (int j = 0; j < n; j++) {
                int input = sc.nextInt();
                xi.add(input);
                sum = sum + input;
            }
            cases.put(xi, sum);
        }
        for (List<Integer> p : cases.keySet()) {
            dp(p, cases.get(p));
        }
    }

    static void dp(List<Integer> nums, int sum) {
        if (nums.size() % 2 != 0) {
            nums.add(0);
        }
        int size = nums.size();
        int tar = sum / 2 + 1;
        int[][][] dp = new int[size + 1][size / 2 + 1][tar + 1];
        for (int i = 1; i <= size; i++) {
            for (int j = 1; j <= size / 2; j++) {
                for (int k = 1; k <= tar; k++) {
                    dp[i][j][k] = dp[i - 1][j][k];
                }
                for (int s = tar; s >= nums.get(i - 1); s--) {
                    dp[i][j][s] = Math
                        .max(dp[i - 1][j - 1][s - nums.get(i - 1)] + nums.get(i - 1),
                            dp[i - 1][j][s]);
                }
            }
        }
        int res = dp[size][size / 2][tar];
        if (res > sum - res) {
            System.out.println((sum - res) + " " + res);

        } else {
            System.out.println(res + " " + (sum - res));
        }
    }

    public static void main11(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        Map<Integer, Integer> input = new HashMap();
        int max = Integer.MIN_VALUE;
        int M = 0;
        for (int i = 0; i < n; i++) {
            int x = sc.nextInt();
            int y = sc.nextInt();
            input.put(y, x);
            M = x + M;
        }
        List<Integer> keySet = new ArrayList<>(input.keySet());
        keySet.sort(Comparator.comparingInt(o -> o));
        int t = M / 2 - 1;
        while (t > 0) {
            int i = 0;
            int j = keySet.size() - 1;
            if (input.get(keySet.get(i)) == 0) {
                i++;
                continue;
            }
            if (input.get(keySet.get(j)) == 0) {
                j--;
                continue;
            }
            int a = keySet.get(i);
            int b = keySet.get(j);
            max = Math.max(max, a + b);
            input.put(a, input.get(a) - 1);
            input.put(b, input.get(b) - 1);
            t--;
        }
        System.out.print(max);
    }

    public static void main9(String[] args) {

        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        sc.nextLine();
        String[] input = new String[t];
        for (int i = 0; i < t; i++) {
            String l = sc.nextLine();
            input[i] = sc.nextLine();
        }
        for (String s : input) {
            if (s.length() < 11) {
                System.out.println("NO");
            } else {
                int index = s.indexOf("8");
                if (index == -1 || s.length() - index < 10) {
                    System.out.println("NO");
                } else {
                    System.out.println("YES");
                }
            }
        }
    }

    private static String minPali(String s) {

        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0, end = 0;
        int r = 0;
        for (int i = 0; i < s.length(); i++) {
            int r1 = expandAroundCenter(s, i, i);
            int r2 = expandAroundCenter(s, i, i + 1);
            int r3 = Math.max(r1, r2);
            r = Math.max(r, r3);
            if (i > s.length() / 2) {
                break;
            }
        }
        StringBuilder rest = new StringBuilder(s.substring(r));
        rest.reverse();
        rest.append(s);

        return rest.toString();
    }

    private static int expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        if (L == -1) {
            return R;
        } else {
            return -1;
        }
    }

    public static boolean isSqure(Pair<Integer, Integer>[] points) {
        Map<Integer, Integer> distances = new HashMap<>();
        for (int i = 1; i <= 4; i++) {
            for (int j = 1; j <= 4; j++) {
                if (i == j || distances.containsKey(i * j)) {
                    continue;
                }
                distances.put(i * j, getDistance(points[i - 1], points[j - 1]));
            }
        }
        List<Integer> disList = new ArrayList<>(distances.values());
        disList.sort(Comparator.comparingInt(o -> o));
        int count1 = 0;
        int count2 = 0;
        for (int dis : disList) {
            if (dis == disList.get(0)) {
                count1++;
            }
            if (dis == disList.get(4)) {
                count2++;
            }
        }
        return count1 == 4 && count2 == 2;
    }

    public static int getDistance(Pair<Integer, Integer> p1, Pair<Integer, Integer> p2) {
        return (p2.getKey() - p1.getKey()) * (p2.getKey() - p1.getKey()) +
            (p2.getValue() - p1.getValue()) * (p2.getValue() - p1.getValue());
    }

    public static void main2(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int m = in.nextInt();
        long[][] dp = new long[m + n][n + 1];
        for (int i = 0; i < m + n; i++) {
            for (int j = 0; j <= n; j++) {
                dp[i][j] = -1;
            }
        }
        System.out.println(C(m + n - 1, m, dp) % 1000000007);
    }

    private static long C(int m, int n, long[][] dp) {
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return m;
        }
        if (n > m / 2) {
            return C(m, m - n, dp) % 1000000007;
        }
        if (n > 1) {
            if (dp[m][n] == -1) {
                dp[m][n] = C(m - 1, n - 1, dp) % 1000000007 + C(m - 1, n, dp) % 1000000007;
            }
            return dp[m][n] % 1000000007;
        }
        return -1;
    }

    public static void main1(String[] args) {
        Scanner in = new Scanner(System.in);
        List<Integer> input = new ArrayList<>();
        while (in.hasNextInt()) {
            input.add(in.nextInt());
        }
        int len = input.size();
        if (len <= 1) {
            System.out.println(0);
        } else {
            int[] dp = new int[len + 1];
            for (int i = 1; i <= len; i++) {
                dp[i] = i < len / 2 ? 1 : -1;
            }
            findMinSteps(input, len, dp);
            System.out.println(dp[len]);
        }
    }

    static int findMinSteps(List<Integer> input, int cur, int[] dp) {
        int res = Integer.MAX_VALUE;
        for (int i = 1; i < cur; i++) {
            if (input.get(i) + i == cur - 1) {
                res = Math
                    .min(res, dp[i] != -1 ? (dp[i] + 1) : (findMinSteps(input, i + 1, dp) + 1));
            }
        }
        if (res == Integer.MAX_VALUE) {
            dp[cur] = -1;
        } else {
            dp[cur] = res;
        }
        return dp[cur];
    }


}
