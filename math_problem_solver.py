"""
數學問題求解器：找出所有符合條件的正整數組合 (a, p)

問題：找出所有符合條件的正整數組合 (a, p)，其中 p 為質數，使得 p^a + a^4 是一個完全平方數。

解題思路：
1. 對於每個正整數 a，我們需要找到質數 p，使得 p^a + a^4 是完全平方數
2. 設 p^a + a^4 = k^2，其中 k 是某個正整數
3. 則 p^a = k^2 - a^4 = (k + a^2)(k - a^2)
4. 由於 p 是質數，且 p^a 是 p 的冪次，所以 (k + a^2) 和 (k - a^2) 都必須是 p 的冪次
5. 設 k + a^2 = p^b，k - a^2 = p^c，其中 b > c ≥ 0，且 b + c = a
6. 則 2a^2 = p^b - p^c = p^c(p^(b-c) - 1)
7. 這給了我們一個限制條件來尋找解
"""

import math
from typing import List, Tuple
import time

def is_prime(n: int) -> bool:
    """
    判斷一個數是否為質數
    
    Args:
        n: 要檢查的數
        
    Returns:
        如果是質數返回True，否則返回False
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # 只需要檢查到 sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_perfect_square(n: int) -> bool:
    """
    判斷一個數是否為完全平方數
    
    Args:
        n: 要檢查的數
        
    Returns:
        如果是完全平方數返回True，否則返回False
    """
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n

def get_perfect_square_root(n: int) -> int:
    """
    如果 n 是完全平方數，返回其平方根；否則返回 -1
    
    Args:
        n: 要檢查的數
        
    Returns:
        如果是完全平方數返回平方根，否則返回 -1
    """
    if n < 0:
        return -1
    root = int(math.sqrt(n))
    if root * root == n:
        return root
    return -1

def find_solutions_method1(max_a: int = 10, max_p: int = 100) -> List[Tuple[int, int]]:
    """
    方法1：直接暴力搜尋
    
    Args:
        max_a: a 的最大值
        max_p: p 的最大值
        
    Returns:
        符合條件的 (a, p) 組合列表
    """
    solutions = []
    
    print(f"Method 1: Searching for a <= {max_a}, p <= {max_p}")
    
    for a in range(1, max_a + 1):
        a4 = a ** 4
        print(f"Checking a = {a}, a^4 = {a4}")
        
        for p in range(2, max_p + 1):
            if not is_prime(p):
                continue
                
            pa = p ** a
            result = pa + a4
            
            if is_perfect_square(result):
                k = get_perfect_square_root(result)
                solutions.append((a, p))
                print(f"  Found solution: (a={a}, p={p}), p^a + a^4 = {pa} + {a4} = {result} = {k}^2")
    
    return solutions

def find_solutions_method2(max_a: int = 20) -> List[Tuple[int, int]]:
    """
    方法2：使用數學分析
    
    根據 p^a + a^4 = k^2 和 2a^2 = p^c(p^(b-c) - 1) 的關係
    
    Args:
        max_a: a 的最大值
        
    Returns:
        符合條件的 (a, p) 組合列表
    """
    solutions = []
    
    print(f"Method 2: Mathematical analysis for a <= {max_a}")
    
    for a in range(1, max_a + 1):
        a4 = a ** 4
        two_a_squared = 2 * a * a
        
        print(f"Checking a = {a}, 2a^2 = {two_a_squared}")
        
        # 對於每個可能的 c 值
        for c in range(a + 1):  # c 可以從 0 到 a
            b = a - c  # b + c = a
            
            if b <= c:  # 需要 b > c
                continue
                
            # 計算 p^c 和 p^(b-c) - 1
            # 2a^2 = p^c * (p^(b-c) - 1)
            # 所以 p^c 必須整除 2a^2
            
            # 尋找可能的 p 值
            for p in range(2, min(two_a_squared + 1, 1000)):
                if not is_prime(p):
                    continue
                    
                # 檢查 p^c 是否整除 2a^2
                pc = p ** c
                if two_a_squared % pc != 0:
                    continue
                    
                # 計算 p^(b-c) - 1
                pb_minus_c = p ** (b - c)
                if pb_minus_c - 1 == two_a_squared // pc:
                    # 驗證解
                    pa = p ** a
                    result = pa + a4
                    if is_perfect_square(result):
                        k = get_perfect_square_root(result)
                        solutions.append((a, p))
                        print(f"  Found solution: (a={a}, p={p}), p^a + a^4 = {pa} + {a4} = {result} = {k}^2")
                        print(f"    c={c}, b={b}, p^c={pc}, p^(b-c)={pb_minus_c}")
    
    return solutions

def find_solutions_method3(max_a: int = 30) -> List[Tuple[int, int]]:
    """
    方法3：更高效的搜尋
    
    注意到 p^a + a^4 = k^2，所以 k^2 > a^4，即 k > a^2
    設 k = a^2 + m，其中 m > 0
    則 p^a + a^4 = (a^2 + m)^2 = a^4 + 2a^2*m + m^2
    所以 p^a = 2a^2*m + m^2 = m(2a^2 + m)
    
    Args:
        max_a: a 的最大值
        
    Returns:
        符合條件的 (a, p) 組合列表
    """
    solutions = []
    
    print(f"Method 3: Efficient search for a <= {max_a}")
    
    for a in range(1, max_a + 1):
        a4 = a ** 4
        two_a_squared = 2 * a * a
        
        print(f"Checking a = {a}, a^4 = {a4}")
        
        # 對於每個可能的 m 值
        max_m = min(1000, a4)  # 限制 m 的範圍以避免過度計算
        
        for m in range(1, max_m + 1):
            pa_value = m * (two_a_squared + m)
            
            # 檢查 pa_value 是否為某個質數的 a 次方
            if pa_value <= 0:
                continue
                
            # 嘗試找到 p，使得 p^a = pa_value
            p_candidate = int(round(pa_value ** (1.0 / a)))
            
            # 驗證 p_candidate 是否為質數，且 p_candidate^a = pa_value
            if (p_candidate ** a == pa_value) and is_prime(p_candidate):
                result = pa_value + a4
                k = get_perfect_square_root(result)
                if k != -1:
                    solutions.append((a, p_candidate))
                    print(f"  Found solution: (a={a}, p={p_candidate}), p^a + a^4 = {pa_value} + {a4} = {result} = {k}^2")
                    print(f"    m={m}, k=a^2+m={a*a}+{m}={k}")
    
    return solutions

def verify_solution(a: int, p: int) -> bool:
    """
    驗證解是否正確
    
    Args:
        a: 正整數 a
        p: 質數 p
        
    Returns:
        如果 p^a + a^4 是完全平方數返回True，否則返回False
    """
    pa = p ** a
    a4 = a ** 4
    result = pa + a4
    
    if is_perfect_square(result):
        k = get_perfect_square_root(result)
        print(f"Verification: p^a + a^4 = {p}^{a} + {a}^4 = {pa} + {a4} = {result} = {k}^2 ✓")
        return True
    else:
        print(f"Verification failed: p^a + a^4 = {p}^{a} + {a}^4 = {pa} + {a4} = {result} ✗")
        return False

def main():
    """主函數"""
    print("="*60)
    print("數學問題求解器：找出所有符合條件的正整數組合 (a, p)")
    print("條件：p 為質數，使得 p^a + a^4 是一個完全平方數")
    print("="*60)
    
    start_time = time.time()
    
    # 使用三種方法尋找解
    print("\n1. 使用直接暴力搜尋方法：")
    solutions1 = find_solutions_method1(max_a=10, max_p=100)
    
    print("\n2. 使用數學分析方法：")
    solutions2 = find_solutions_method2(max_a=15)
    
    print("\n3. 使用高效搜尋方法：")
    solutions3 = find_solutions_method3(max_a=20)
    
    # 合併所有解並去重
    all_solutions = list(set(solutions1 + solutions2 + solutions3))
    all_solutions.sort()
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print("最終結果")
    print("="*60)
    
    if all_solutions:
        print(f"找到 {len(all_solutions)} 個解：")
        for i, (a, p) in enumerate(all_solutions, 1):
            print(f"{i}. (a={a}, p={p})")
            
            # 驗證解
            pa = p ** a
            a4 = a ** 4
            result = pa + a4
            k = get_perfect_square_root(result)
            print(f"   驗證：p^a + a^4 = {p}^{a} + {a}^4 = {pa} + {a4} = {result} = {k}^2")
    else:
        print("未找到符合條件的解")
    
    print(f"\n計算時間：{end_time - start_time:.2f} 秒")
    
    # 分析結果
    if all_solutions:
        print("\n" + "="*60)
        print("結果分析")
        print("="*60)
        
        a_values = [sol[0] for sol in all_solutions]
        p_values = [sol[1] for sol in all_solutions]
        
        print(f"a 的範圍：{min(a_values)} 到 {max(a_values)}")
        print(f"p 的範圍：{min(p_values)} 到 {max(p_values)}")
        print(f"不同的 a 值：{sorted(set(a_values))}")
        print(f"不同的 p 值：{sorted(set(p_values))}")
        
        # 檢查是否有模式
        print("\n可能的模式分析：")
        for a in sorted(set(a_values)):
            p_for_a = [p for a_val, p in all_solutions if a_val == a]
            print(f"  當 a = {a} 時，p 的值：{sorted(p_for_a)}")

if __name__ == "__main__":
    main() 