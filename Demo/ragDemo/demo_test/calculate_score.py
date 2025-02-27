# edit distance
def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    # Initialize the DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                               dp[i][j - 1] + 1,      # Insertion
                               dp[i - 1][j - 1] + 1)  # Substitution
    
    return dp[m][n]

if __name__ == '__main__':
    # Example usage
    str1 = "kitten"
    str2 = "sitting"
    print(f"Edit distance between '{str1}' and '{str2}' is {edit_distance(str1, str2)}")
