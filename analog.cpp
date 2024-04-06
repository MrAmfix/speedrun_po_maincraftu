#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>
#include <limits>
#include <vector>

using namespace std;

void lab1_task20(){
    ifstream fin('input/lab1/task20.txt');
    ofstream fout('output/lab1/task20.txt');
    int n, k;
    string str;
    fin >> n >> k >> str;
    int c = 0;
    for (int i = 0; i < str.length(); i++) {
	    int m = 0, p = k;
	    while ((i - m) >= 0 && (i + m) < str.length() && p >= 0) {
		    if (str[i - m] == str[i + m]) {
			    c++;
			    m++;
		    }
		    else {
			    p--;
			    m++;
			    if (p >= 0) {
				    c++;
			    }
		    }
	    }
    }
    for (int i = 0; i < str.length() - 1; i++) {
	    int m = 0, p = k;
	    while ((i - m) >= 0 && (i + 1 + m) < str.length() && p >= 0) {
		    if (str[i - m] == str[i + 1 + m]) {
			    c++;
			    m++;
		    }
		    else {
			    p--;
			    m++;
			    if (p >= 0) {
				    c++;
			    }
		    }
	    }
    }
    fout << c;
}

const int MAX_HEIGHT = 5;
const int MAX_MASK = (1 << MAX_HEIGHT) - 1;
bool compatibility_matrix[MAX_MASK + 1][MAX_MASK + 1];
int dp[35][MAX_MASK + 1];
int mask;
int rows, cols;

inline int get_bit(int num, int bit) {
    return (num >> bit) & 1;
}

void precalculate_compatibility_matrix() {
    for (int prev_mask = 0; prev_mask <= mask; ++prev_mask) {
        for (int current_mask = 0; current_mask <= mask; ++current_mask) {
            bool is_compatible = true;
            for (int bit = 0; bit < rows - 1; ++bit) {
                int sum = get_bit(prev_mask, bit) + get_bit(prev_mask, bit + 1) +
                    get_bit(current_mask, bit) + get_bit(current_mask, bit + 1);
                if (sum == 0 || sum == 4) {
                    is_compatible = false;
                    break;
                }
            }
            compatibility_matrix[prev_mask][current_mask] = is_compatible;
        }
    }
}

void lab1_task22() {
    // Тут просто cin, cout
    cin >> rows >> cols;
    if (rows > cols) swap(rows, cols);
    mask = (1 << rows) - 1;
    precalculate_compatibility_matrix();
    for (int mask_value = 0; mask_value <= mask; ++mask_value)
        dp[0][mask_value] = 1;
    for (int k = 1; k < cols; ++k) {
        for (int prev_mask = 0; prev_mask <= mask; ++prev_mask) {
            for (int current_mask = 0; current_mask <= mask; ++current_mask) {
                if (compatibility_matrix[prev_mask][current_mask])
                    dp[k][current_mask] += dp[k - 1][prev_mask];
            }
        }
    }
    int result = 0;
    for (int mask_value = 0; mask_value <= mask; ++mask_value)
        result += dp[cols - 1][mask_value];
    cout << result;
}

vector<vector<int>> floyd_warshall(vector<vector<int>> graph) {
    for (size_t i = 0; i < graph.size(); ++i) {
        for (size_t j = 0; j < graph.size(); ++j) {
            for (size_t k = 0; k < graph.size(); ++k) {
                if (graph[j][i] + graph[i][k] < graph[j][k]) {
                    graph[j][k] = graph[j][i] + graph[i][k];
                }
            }
        }
    }
    return graph;
}

int lab3_task17() {
    int nv, ne;
    cin >> nv >> ne;

    vector<vector<int>> graph(nv, vector<int>(nv, 1000000000));
    for (int i = 0; i < nv; ++i) {
        graph[i][i] = 0;
    }
    for (int i = 0; i < ne; ++i) {
        int s, e;
        cin >> s >> e;
        --s;
        --e;
        graph[s][e] = 0;
        graph[e][s] = min(graph[e][s], 1);
    }

    int ans = 0;
    vector<vector<int>> result = floyd_warshall(graph);
    for (const auto& row : result) {
        for (const auto& elem : row) {
            ans = max(ans, elem);
        }
    }
    cout << ans;

    return 0;
}