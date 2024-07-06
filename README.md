

# Studing algoriths with AlgosWithMichael*
https://www.youtube.com/@AlgosWithMichael

## 1) Dynamic Programming 

Medium
- Time Complexity O(M * N)

Amazon Dynamic Programming Interview Question - Unique Paths
https://www.youtube.com/watch?v=4Zq2Fnd6tl0&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=2

Arobot is located at the top-left corner of a m x n grid (marked "Start" in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked "Finish" in the diagram below)

How many possible unique paths are  there?

Above is a 7 x 3 grid. How many possibleunique paths are thre?

```java
class Solution {
	
	public int uniquePaths(int m, int n) {
		int[][] dp = new int[m][n];
		
		for(int i = 0. i < m; i++) {		
			for(int j = 0; j < n; j++) {			
				if(i == 0 || j == 0) dp[i][j] = 1;
				else dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}		
		}
	
		return dp[m - 1][n - 1];
	
	}
}
```

## 2) Array

- Time Complexity: O(N) We have a total of N digits in our input 'x' and we must perform the calculations for each digit in our input 'x' and we must perform the calculations for each digit in our whilw loop.
- Space Complexity O(1) No extra space!

Technical Interview Question - Reverse Integer [LeetCode]
https://www.youtube.com/watch?v=B-Yc10DUaM8&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=5

Given a 32 bit signed integer, reverse digits os an integer.

Example 1: 

Input: 123
Output: 321

Example 2:

Input: -123
Output -321

Example 3:

Input: 120
Output: 21

Assume we are dealing with an environment which could only store integers within the 32-bit signed integer rage [-231, 231 - 1] (Max number that int can be). For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows

```java
class Solution {
	public int reverse(int x) {
		long result = 0;
		
		while(x != 0) {
			in t remainder = x % 10;
			result = result * 10 + remainder;
			if(result < Integer.MIN_VALUE || result > Integer.MAX_VALUE) return 0;
			x /= 10;
		}
		return (int) result;
	}
}
```
---
Easy

Technical Interview Question: Two Sum - Input Sorted [LeetCode]
https://www.youtube.com/watch?v=zV0hO1heBlE&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=10

Given an array of integers that is already sorted in ascending order, find two numbers such that add up to a specific target number

The function twoSum should return indices of the two numbers such they add up to the target, 
where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based

You may assume that each input would have exactly one solution and you may not use the same element twice.

Input: numbers = {2, 7, 11, 15}, target 9
Output index1 = 1 index2=2

```java
class Solution {
	public int[] twoSum(int[] numbers, int target) {
		int left = 0, right = numbers.length -1;
		
		while(left < right) {
			int sum = numbers[left] + numbers[right];
			if(sum == target) return new int[] {left +1, right +1}
			else if(sum > target) --right;
			else ++left;
		}
		
		return null;
	}
}
```
---
Hard
- Time Complexity O(N) linear - Over all time complexity is 3N but bound sugest linear time complexity
- Space O(1) constant - we don't implement any extra memory in this algorithm

Amazon Coding Interview Question - 41 First Missing Positive (LeetCode)
https://www.youtube.com/watch?v=9SnkdYXNIzM&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=14

Given an unsorted integer array, find the smallest missing positive integer.

Example 1:

Input: [1, 2, 0]
Output: 3

Example 2:

Input: [3, 4, -1, 1]
Output: 2

Example 3:

Input: [7, 8, 9, 11, 12]
Output: 1

Your algotithm shoud run in O(n) time and uses constant extra space

```java
class Solution {
	public int firstMissingPositive(int[] nums) {
		if(nums == null || nums.length == 0) return 1;
		int n = nums.length, containsOne = 0;
		
		// step 1
		for(int i = 0; i < n; i++) {
			if(nums[i] == 1) {
				containsOne = 1;
			} else if(nums[i] <= 0 || nums[i] > n) {
				nums[i] = 1;
			}
		}
		
		if(containsOne == 0) return 1;
		
		//step2
		for(int i = 0; i < n; i++) {
			int index = Math.abs(nums[i]) - 1;
			if(nums[index] > 0) nums[index] = -1 * nums[index];
		}
		
		//step3
		for(int i = 0; i < n; i++) {
			if(nums[i] > 0) {
				return i + 1;
			}			
		}
		
		return n + 1;
		
	}	
}
```

## 3) To Pointer

Easy
- Time Complexity: 0(N) We must loop throuth each catacter in our string
- Space Complexity: 0(N) We must convert our string to character array in order to do in-place swaps.

Amazon Coding Interview Question - Reverse Vowels In A String [LeetCode]
https://www.youtube.com/watch?v=WAo0mWvHrp8&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=4

Write a function that takes a string as input and reverse only the vowels os a string 

Amazon Coding Interview Question - Reverse Vowels In A String [LeetCode]

Example 1:

Input: "hello"
Output: "holle"

Example 2:

Input: "leetcode"
Output: "leotcede"

Note: The cowels does not include the letter "y".

```java
class Solution {
	public String reverseVowels(String s) {
		char[] arr = s.toCharArray();
		int left = 0, right = arr.length - 1;'
		
		while(left < right) {
			boolean leftIsVowel(arr[left]);
			boolean rightIsVowel(arr[right]);
			
			if(leftIsVoewl && rightIsVowel) {
				swap(arr, left, right);
				++left;
				--right;
			}
			
			if(!leftIsVowel) {
				++left;
			}
			
			if(!rightIsVowel) {
				--right;
			}
			
		}
		return new String(arr);
	}
	
	private void swap(char[] arr, int i, int j) {
		char temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
	
	private boolean isVowel(char letter) {
		
		char c = Character.toLowerCase(letter);
		
		return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';

	}
}
```

## 4) Depth (Island)

Medium
- Technical Interview Question: Number of Islands [LeetCode]
- https://www.youtube.com/watch?v=CLvNe-8-6s8

Given 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontaly or vertically, You nmay assume all four edges of the grid are all surrounded by water

Example 1: 

11110
11010
11000
00000 

Answer 1

Example 2:

11000
11000
00100
00011

Answer 3

0 = water
1 = land
islandCount = 0

```java
class Solution {

	public int numIslands(char[] grid) {
		
		if(grid == null) return 0;
		
		int islandCount = 0;
		
		for(int i = 0; i < grid.length; i++) {
			for(int j = 0; j < grid[0].length; j++) {
				if(grid[i][j] == '1') {
					//increase our island count
					++islandCount;
										
					// change any other land connected to zeros
					changeLandToWater(grid, i, j);				
				}
			}
		}
		return islandCount;	
	}
	
	private void changeLandToWater(char[][] grid, int i, int j) {
		// 1) row less 0
		// 2) row greater than grid.length (row length)
		// 3) column less 0
		// 4) column greater than grid[0].length (column length)
		// 5) if position is a '0'
		
		if( i < 0 || i >= hrid.length || j < 0 || j >= grid[0].length || grid[i][j] == '0') return;
		
		// we know thr position must be a '1'
		grid[i][j] = '0';
		
		changeLandToWater(grid, i - 1, j); // down
		changeLandToWater(grid, i + 1, j); // up
		changeLandToWater(grid, i, j - 1); // left
		changeLandToWater(grid, i, j + 1); // right
				
	}
	
}
```
---
Medium
- Time Complexity: O(M * N) we must interate over every element in our grid
- Space Complexity: O(M * N) Our recursion depth could potentially be M * N

Amazon Coding Interview Question - Number of Distinct Islands
https://www.youtube.com/watch?v=c1ZxUOHlulo&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=3

Given a non-empty 2D array grid of 0`s and 1`s an island is a group of 1`s (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands. An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.

Example 1:

11000
11000
00011
00011

Given the above grid map, return 1

Example 2:

11011
10000
00001
11011

Given the above grid map, return 3.

```java
class Solution {
	// X = start
	// O = out of bounds OR water
	// U = up
	// D = down
	// R = right
	// L = left
	public int numDistinctIslands(int[][] grid) {
		if(grid == null || grid.length == 0) return 0;
		
		Set<String> set = new HashSet<>();
		
		int m = grid.length, n = grid[0].length;
		
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				if(grid[i][j] == 1) {
					String path = computePath(grid, i, j, m, n, "X");
					set.add(path);
				}
			}
		}
		return set.size();
	
	}

	private String computePath(int[][] grid, int i, int j, int m, int n, String direction) {
		
		if(i < 0 || j < 0 || i >= m || j >= m || grid[i][j] == 0) return "0";
		greid[i][j] = 0;
		String left = computePath(grid, i, j - 1, m, n, "L");
		String right = computePath(grid, i, j + 1, m, n, "R");
		String up = computePath(grid, i - 1, j, m, n, "U");
		String down = computePath(grid, i + 1, j, m, n, "D");
		return direction + left + right + up+ down;
	}

}	
```
---

Medium
- Time Complexity : 0(N * M) N is the number of rowns we have, M is the number of columns. We interate over the entire grid rouching each square a single time.
- Space Complexity: O(N * M) In the worst case, we will have a recursion depth of N * M in the case where we have ALL 1's in our grid.

Technical Interview Question - Max Area of Island [LeetCode]
https://www.youtube.com/watch?v=JP39wU1UhRs&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=6

Given a non-empty grid of 0's and 1's and island is a group of 1 's (representing land) connected 4-directionally (horizontal or vertical) You may assume all four edges of the grid are surrounded by water.

Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)

Example:
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]
]

Given  the above grid return 6. Note the answer is not 11 because the island must be connected 4-directionally.

Example:
[[0,0,0,0,0,0,0,0]]

Given the above grid return 0.

Note: The length of each dimension in the given grid does not exceed 50.

```java
class Solution {
	public int maxAreaOfIsland(int[][] grid) {
		int max = 0, int n = gti.length, int m grid[0].length;
		for(int i = 0; i < n; i++) {
			for(int j = 0, j < m; j++) {
				if(grid[i][j] == 1) {
					int area = getArea(grid, i, j, n, m);
					max = Math.max(max, area);
				}
			}
		}
		return max;
	}	
	private int getArea(int[][] grid, int i, int j, int n, int m) {
		// 1) check for out of bounds
		// 2) check is the position is water ('0')
		if(i < 0 || j < 0 || i >= m || j >= m || grid[i][j] == 0) return 0;
		
		// we know that we hava a number 1
		// 1) check left, up, right, down for 1's
		// 2) change the '1' to a '0'
		grid[i][j] = 0;
		int left = getArea(grid, i, j  - 1, n, m);
		int right = getArea(grid, i, j  + 1, n, m);
		int up = getArea(grid, i - 1, j, n, m);
		int down = getArea(grid, i + 1, j, n, m);
		
		return left + right + up + down + 1;		
	}	
}
```
## 5) Remove Duplicates

Medium
- Time Complexity O(M * N) - We must touch each element a single time
- Space Compexity O(1) - If you consider the result as extra space, it would be O(N)

Amazon Coding Interview Question - Find All Duplicates in Array [LeetCode]
https://www.youtube.com/watch?v=lYxEdtR5_xQ&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=2

Given an array integers, 1 <= a[i] <= n (n = size of array), some elements appear twicw and others appear once

Find all the elements that appear twice in this array.

Could you do it without extra space and in O(n)runtime?

Example:
[4,3,2,4,8,2,3,1]

Output:
[2, 3]

```java
class Solution {
	public List<Integer> findDuplicates(int[] nums) {
		List<Integer> result = new ArrayList<>();
		for(int = 0; i < nums.length; i++){
			int index = Math.abs(nums[i]) - 1;
			
			if(nums[index] < 0) {
				result.add(index + 1);
			}
			
			nums[index] = -nums[index];
		}
		
		return result;	
	}
}
```
## 6 STRING

Easy
Technical Interview Question: First Unique Character in a String [LeetCode]
https://www.youtube.com/watch?v=BJGNVQiLNDs&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=7

Given a string, find the fist non-repeating character in it and return it's index. If it doesn't exist, return -1

Examples:
s = "leetcode"
return 0.

s = "leveleetcode",
return 2.

Note: You may assume the string contain only lowercase letters

```java
class Solution {

	// 1st step: Getting a count of all characters in the string
	// 2nd step: Compare input string characters to our key/value pairs
	public int fisrtUniqChar(String s) {
		// Step 1
		Map<Character, Integer> countMap = new HashMap<>();
		
		char letter = s.charAt(i);
		
		for(int i = 0; i < s.length(); i++) {
			if(countMap.containsKey(letter)) {
				countMap.put(letter, conuntMap.get(letter) + 1);
			} else {
				countMap.put(letter, 1);
			}
		}
		
		// Step 1
		for(int i = 0; i < s.length(); i++) {
			char letter = s.charAt(i);
			
			if(countMap.get(letter) == 1) {
				return i;
			}
		
		}
		return -1;
	}
}
```

## 7 Valid Anagram

Easy
Technical Interview Question: 242. Valid Anagram [LeetCode]
https://www.youtube.com/watch?v=ikO4qKG_IWc&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=8

Given two strings s and t, write a sinction to determine if t is an anagram os s.

For example:
s = "anagram, t = "nagaram" return true
s = "rat", t = "car", return false


Note: You may assume the string contains only lowercase alphabets
Follow up: What if the inputs contain unicode characters? How wold you adapt your solution to such case?

```java
//arr1[0] = 'a' - 'a' = 0
class Solution {
	public boolean isAnagram(String s, String t) {
		int[] letters = new int[26];
		char[] arr1 = s.toCharArray();
		char[] arr2 = t.toCharArray();
		
		for(int i = 0; i < arr1.length; i++){			
			letters[arr1[i] - 'a']++;
		}
		
		for(int i = 0; i < arr1.length; i++){
			letters[arr2[i] - 'a']--;
		}
		
		for(int letter : letters) {
			if(letter != 0) {
				return false;
			}			
		}		
		return true;
	}
}
```

## 8 Deph-fist (DFS) 

Medium
- Time complexity O(N) - N igual the number of Nodes;
- Space: O(N) - N igual the number of Nodes;

Amazon Coding Interview Question - 133 Clone Graph (LeetCode)
https://www.youtube.com/watch?v=e5tNvT1iUXs&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=11

Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a val (int) and a list (List[Node]) of its neighbors.

class Node {
	public int val;
	public List<Node> neighbors;
}

Test case format:

For simplicity sake each node's value is the same as the node's index(1-indexed). For example the 
fist node with val = 1, the second node with val = 2, and so on. The graph is represented in test case using an adjacency list.

Adjacency list is a collection of unordered lists used to represent a finite graph, Each list describe the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the give node as a reference to the cloned graph.


1 - 2
3 - 4

Conections
1, [2, 3] 
2, [1, 4]
3, [1, 4]
4, [2, 3]

(You don't have any part of graph separated)

Map<int, Node>

1 -> Node[1]
  [2, 3]
2 -> Node[2]
  [1, 4]
4 -> Node[4]
  [2, 3]
3 -> Node[3]
  [1, 4]
  
```java  
/*

class Node {
	public int val;
	public List<Node> neighbors;
	
	public Node() {
		val = 0;
		neighbors = new ArrayList<Node>();
	}
	
	public Node (int _val) {
		val = _val;
		neighbors = new ArrayList<Node>();
	}
	
	public Node (int _val, ArrayList<Node> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}
	
}

*/

class Solution {
	public Node cloneGraph(Node node) {
		
		if(node == null) return null;
		
		Map<Integer, Node> map = new HashMap<>();
		return cloneGraph(node, map);
	}
	
	private Node cloneGraph(Node node, Map<Integer, Node> map) {
		if(map.containsKey(node.val)) return map.get(node.val);
		
		Node copy = new Node(node.val);
		map.put(node.val, copy)
		for(Node neighbor, node.neighbors) copy.neighbors.add(cloneGraph(neighbor, map));
		return copy;
	}
}
```

## 9  Recursive 

Easy
- Time Complexity O(N) - 'N' the number of nodes that you have in your binary tree
- Space Complexity O(N) - 'N' the number of nodes that you have in your binary tree

Amazon Coding Interview Question - 404. Sum of Left Leaves (LeetCode)
https://www.youtube.com/watch?v=DSqSHwDE82M&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=12

Find the sum of all left leaves in a given binary tree.

Example:

     3
    / \
   9   20
      /  \
     15   7
     
There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24 

```java
/*
Definition for a binary tree node.

public class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;
	TreeNode (int x) { val = x; }
}
*/
class Solution {
	public int sumOfLeftLeaves(TreeNode root) {
		if(root == null) return 0;
		int sum = 0;
		
		if(root.left != null) {
			if(isLeaf(root.left)) sum += root.left.val;
			else sum += sumOfLeftLeaves(root.left);
		}
		sum += sumOfLeftLeaves(root.right);
		return sum;		
	}
	
	private boolean isLeaf(TreeNode node) {
		return node left == null && node.right == null;
	}

}
```
Outher Solution using "Queue"

```java
class Solution {
	public int sumOfLeftLeaves(TreeNode root) {
		if(root == null) return 0;
		int sum = 0;
		
		Queue<TreeNode> queue = new LinkedList<>();
		queue.add(root); 	
		while(!queue.isEmpty) {
			TreeNode node = queue.poll();
			if(root.left != null) {
				if(isLeaf(root.left)) sum += root.left.val;
				else queue.add(node.left);
			}
			if(node.right != null) queue.add(node.right);
			
		}
		return sum;
	}
	
	private boolean isLeaf(TreeNode node) {
		return node left == null && node.right == null;
	}
}
```

## 10 Stack 

Easy
- Time Complexity: O(1) becase you are using a "Stack";
- Space Complexity: O(N) number of elements that you adding

Amazon Coding Interview Question - 155. Min Stack (LeetCode)
https://www.youtube.com/watch?v=3hd7zLNesaE&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=13

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

- push(x) -- Push element x onto stack.
- pop() -- Removes the element on top of the stack.
- top() -- Get the top element.
- getMin() -- Retrieve the minimum element in the stack.

Example:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); --> Return -3.

minStack.pop();
minStack.top(); --> Return 0.

minStack.getMin(); --> Return -2.

```java
/*
Your Min Stack object will be instantiated and called as such:
MinStack obj = new MinStack();
obj.push(x);
obj.pop();
int param_3 = obj.top();
int param_4 = obj.getMin();
*/
class MinStack {
	
	private Stack<Integer> s;
	private Stack<Integer> t;
	
	
	
	public MinStack() {
		s = new Stack<>();
		t = new Stack<>();
	}
	
	public void push(int x) {
		s.push(x);
		int min = t.isEmpty || x < t.peek() ? x : t.peek();
		t.push(min);
	}
	
	public void pop() {
		s.pop();
		t.pop();
	}
	
	public int top() {
		return s.peek();
	}
	
	public int getMin() {
		return t.peek();
	}

}
```
## 11 Matrix (rotate)  

Medium
- Time: O(N²)
- Space: O(1) requirement is constant space

Microsoft Coding Interview Question - Rotate Image
https://www.youtube.com/watch?v=J-Ihez5cwCM&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=15

You are given an n x n 2D matrix representing an image

Rotate the image by 90 degres (clockwise)

Note:

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1: 

Given input matrix =
```
[
	[1,2,3],
	[4,5 6],
	[7,8,9]
]
```

rotate the input matrix in-place suth that it becomes:
```
[
	[7,4,1],
	[8,5,2],
	[9,6,3]
]
```
```java
class Solution {
	public void rotate(int[][] matrix) {
		if(matrix == null || matrix.length == 0) return;
	}
	
	private void tranposeMatrix(int[][] matrix, int n) {
		for(int i = 0; i < n; i++) {
			for(int j = i; j < n; j++) {
				swap(matrix, i, j, j, i); //[i][j] <-> [j][i]
			}
		}
	}
	
	private void reverseMatrix(int[][] matrix, int n) {
		for(int i = 0; i < n; i++) {
			for(int j = 0, k = n - 1; j < k; --k) {
				swap(matrix, i, j, i, k); //[i][j] <-> [i][k]
			}
		}
	}
	
	private void swap (int[][] matrix, int i, int j, int k, int l) {
		int temp = matrix[i][j];
		matrix[i][j] = matrix[k][l];
		matrix[k][l] = temp;
	}
}
```
## 12 Binary Tree

Medium
- Time (O(N) you have to loop over all your nodes in your tree
- Space O(N) 

FACEBOOK CODING INTERVIEW QUESTION - 236 LOWEST COMMON ANCESTOR
https://www.youtube.com/watch?v=xuvw11Ucqs8&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=16

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree

According to the definition of LCA on Wikipedia? The lowest common ancestor is defined between  two nodes p and q as the lowest node in T that has toth p and q as descendants (where we allow a node to be a descendant of itself)".

Given the following binary tree: root = [3,5,1,6,2,0,8,null, null,7,4]
```
      3
    /   \
   5      1
  / \    / \ 
 6   2  0   8
    / \
   7   4	
```

posturder
Left
Right
Visit

//Comom ancester for both

```java
/*
Definition for a binary tree node
public class TreeNode{
	int val;
	TreeNode left;
	TreeNode right;
	TreeNode(int x) { val = x; }
}
*/
class Solution {

	private TreeNode result;

	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		findLCA(root, p, q);
		return result;
	}
	
	private boolean findLCA(TreeNode root, TreeNode p, TreeNode q) {
		if(root == null) return false;
		boolean left = findLCA(root.left, p, q);
		boolean right = findLDA(foot.right, p, q);
		boolean curr = root == p || root == q;
		if((left && right) || (left && curr) || (right && curr)) result = root;
		return left || right || curr;
		
	}
}
```
---

Easy
- Time O(N)
- Space O(N)

FACEBOOK CODING QUESTION - 455 ADD STRINGS (LEETCODE)
https://www.youtube.com/watch?v=PlCOTfHB54g&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=17

Given two non-negative num1 and num2 represented as string, return the sum of num1 and num2.

Note:

1. The lengh of both num1 and num2 is < 5100.
2. Both num1 and num2 contains only digits 0-9.
3. Both num1 and num2 does not contain any leading zero.
4. You must not use any built-in BigInteger library or convert the inputs to Integer directly.
         
     i   
 "2859"
    j
 "293"

'9' - '0' = 9
'3' - '0' = 3

9 + 3 = 12
12/10 = 1
12%10 = 2
carry = 1
result = 25

****
Use ASC Table

48 - 0
49 - 1
50 - 2
51 - 3
52 - 4
53 - 5
54 - 6
55 - 7
56 - 8
57 - 9

'1' - '0' = 1
49 - 48 = 1

```java
class Solution {
	public String addStrings(String num1, String num2) {
		int i = num1.length() -1, j = num2.lenght() -1;
		int carry =0;
		StringBuilder result = new StringBuilder();		
		while(i > -1 || j > -1) {
			int d1 = i > -1 ? num1 ? num1.charAt(i) - '0': 0;
			int d2 = j > - 1 ? num2.charAt(j) - '0' : 0;
			int sum = d1 + d2 +carry;
			result append(sum % 10);
			carry = sum / 10;
			--i --j;			
		}
		// 99 + 1 = 100
		if (carry == 1) result.append(1);
		
		return result.reverse().toString();			
	}
}
```

## 12 Time

Medium
Bucket Sort Interview Question - 539 Min Time Difference (Amazon)
https://www.youtube.com/watch?v=-o_YDXNfRUY&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=18

Given a list of 24-hour cloc time points in Hour:Minutes" format. find the minimum minutes difference between any two time points in the list

Example 1: 

Input: ["23:59", "00:00"]
Output: 1

Note:

1) The number of time points in the given list is at least 2 and won't exceed 20000.
2. The input time is legal and ranges from 00:00 to 23:59.

```java
class Solution {

	public int findMinDiffence(List<String> timePoints) {
	 boolean [] bucket = new boolean[1440]; 
	 
	 for(String timePoint : timePoints) {
	  String [] t = timePoint.sprit(":");
	  int hours = Integer.parse(t[0]);
	  int minutes -= Integer.parse(t[1]);
	  int total = hours * 60 + minutes;
	  bucket[total] = true;
  	}
  	
  	int min = Integer.MAX.VALUE, first = -1, pŕev = -1, curr = -1;
  	for(int i = 0; i < bucket.size(); i++) {
  		if(bucket[i]) {
  			if(prev == -1) {
  				prev = i;
  				first = i 
  			} else {
  				curr = i;
  				min = Math.min(min, curr - prev)
  				prev = curr;
  			}
  		}
  	}  	
  	return Math.min(min, 1440 - curr + firt);
   }
}
```

## 13 Priority Queue

Medium

Top K Frequent Words - 692 Priority Queue Approach (LeetCode)
https://www.youtube.com/watch?v=cupg2TGIkyM&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=19
Given a non empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.

Example 1:i
Input: ["i", "love", "leetcode, "i", "love, coding,"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words. Note that "i" comes before "love" due to a lower alphabetical

Example 2:
Input: ["the", "day", "is", "summy", "the", "the", "the", "the", "sunny"]
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "is", "sunny" and "day" are the four most frequent wprds, with the number of occcurrence being 4, 3, 2 and 1 respectively.

Note:

1. You may assume k is always valid 1 <= k <= number of unique elements.
2. Inout words contain only lowercase letters

Follow up:

Try to solve it in O n log k time and O (n) extra space.
```
	Priority Queue
	/           \
    Min Heap     Max Heap	
 ```

```java
class Solution {
	public List<String> topKFrequent(String[] words, int k) {
		Map<String, Integer> map = new HashMap<>();
		for(String word : words) {
			map.put(word, map.getOrDefault(word, 0) + 1);
		}		
	}
	
	PriorityQueue<String pq = new PriorityQueue<>(new Comparator<String>() 
		@Override
		public int compare(String word1, String word2)
			int frequency1 = map.get(word1);
			int frequency2 = map.get(word2);
			if(frequency1 == frequency2) return word2.compateTo(word1);
			return frequency 1 - frequency 2;
		}
		});		
	for(Map.Entry<String, Integer> entry : map.entreSet()) {
		pq.dd(entry.getKey());
		if(pq.size : k) pq.poll();
	}	
	List<String> resut = new ArrayList<>();
	while (!pq.isEmtry() ) result add(pq.pull())		
}
```
## 13 Linked List

Easy
- Time O(M + N)
- Space O(1)

Coding Interview Question - 21. Merge Two Sorted Lists (LeetCode)
https://www.youtube.com/watch?v=c86I16V8P58&list=PLtQWXpf5JNGJagakc_kBtOH5-gd8btjEW&index=20

Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

Example:

Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4

Head: -1
Curr: null

```java
/*
Definition for sigly-linked list.
public class ListNode {
	int val;
	ListNode next;
	ListNode() {}
	ListNode(int val) { this.val = val; }
	ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}
*/
class Solution {
	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		ListNode head = new ListNode(-1);
		ListNode curr = head;		
		while(l1 != null || l2 != null) {
			if(l1 == null {
				curr.next = l2;
				l2 = l2.next;
			) else if (l2 == null) {
				curr.next = l1;
				l1 = l1.next;
			} else if {l1.val < l2.val) {
				curr.next = l1;
				l1 = l1.next;
			} else {
				//l2.val < l1.val
				curr.next = l2;
				l2 = l2.next;
			}
			cur = curr.next;
		}
		return head.next;				
	}
}
```

