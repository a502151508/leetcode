import java.util.*;
import javafx.util.Pair;


public class test {

    int a;

    public static void main(String[] args) {
        TrieNode t = new TrieNode();

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
