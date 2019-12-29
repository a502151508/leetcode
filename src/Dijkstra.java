/**
 * Created by Sichi Zhang on 2019/11/26.
 */

/*
指定一个节点，例如我们要计算 'A' 到其他节点的最短路径
引入两个集合（S、U），S集合包含已求出的最短路径的点（以及相应的最短长度），U集合包含未求出最短路径的点（以及A到该点的路径，注意 如上图所示，A->C由于没有直接相连 初始时为∞）
初始化两个集合，S集合初始时 只有当前要计算的节点，A->A = 0，
U集合初始时为 A->B = 4, A->C = ∞, A->D = 2, A->E = ∞，敲黑板！！！接下来要进行核心两步骤了
从U集合中找出路径最短的点，加入S集合，例如 A->D = 2
更新U集合路径，if ( 'D 到 B,C,E 的距离' + 'AD 距离' < 'A 到 B,C,E 的距离' ) 则更新U
循环执行 4、5 两步骤，直至遍历结束，得到A 到其他节点的最短路径

O(N^2)
 */
public class Dijkstra {

    public static final int M = 10000; // 代表正无穷

    public static void main(String[] args) {
        // 二维数组每一行分别是 A、B、C、D、E 各点到其余点的距离,
        // A -> A 距离为0, 常量M 为正无穷
        int[][] weight1 = {
            {0, 4, M, 2, M},
            {4, 0, 4, 1, M},
            {M, 4, 0, 1, 3},
            {2, 1, 1, 0, 7},
            {M, M, 3, 7, 0}
        };

        int start = 0;

        int[] shortPath = dijkstra(weight1, start);

        for (int i = 0; i < shortPath.length; i++) {
            System.out.println("从" + start + "出发到" + i + "的最短距离为：" + shortPath[i]);
        }
    }

    public static int[] dijkstra(int[][] weight, int start) {
        // 接受一个有向图的权重矩阵，和一个起点编号start（从0编号，顶点存在数组中）
        // 返回一个int[] 数组，表示从start到它的最短路径长度
        int n = weight.length; // 顶点个数
        int[] shortPath = new int[n]; // 保存start到其他各点的最短路径
        String[] path = new String[n]; // 保存start到其他各点最短路径的字符串表示
        for (int i = 0; i < n; i++) {
            path[i] = start + "-->" + i;
        }
        int[] visited = new int[n]; // 标记当前该顶点的最短路径是否已经求出,1表示已求出

        // 初始化，第一个顶点已经求出
        shortPath[start] = 0;
        visited[start] = 1;

        for (int count = 1; count < n; count++) { // 要加入n-1个顶点
            int k = -1; // 选出一个距离初始顶点start最近的未标记顶点
            int dmin = Integer.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                if (visited[i] == 0 && weight[start][i] < dmin) {
                    dmin = weight[start][i];
                    k = i;
                }
            }

            // 将新选出的顶点标记为已求出最短路径，且到start的最短路径就是dmin
            shortPath[k] = dmin;
            visited[k] = 1;

            // 以k为中间点，修正从start到未访问各点的距离
            for (int i = 0; i < n; i++) {
                //如果 '起始点到当前点距离' + '当前点到某点距离' < '起始点到某点距离', 则更新
                if (visited[i] == 0 && weight[start][k] + weight[k][i] < weight[start][i]) {
                    weight[start][i] = weight[start][k] + weight[k][i];
                    path[i] = path[k] + "-->" + i;
                }
            }
        }
        for (int i = 0; i < n; i++) {

            System.out.println("从" + start + "出发到" + i + "的最短路径为：" + path[i]);
        }
        System.out.println("=====================================");
        return shortPath;
    }
}
