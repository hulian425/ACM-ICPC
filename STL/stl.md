
## 二分查找
1. **lower_bound**
	(起始地址，结束地址，要查找的数值) 返回的是数值 第一个 出现的**位置**。
（	返回大于等于val的值）


2. **upper_bound**
(起始地址，结束地址，要查找的数值) 返回的是数值 最后一个 出现的**位置**。
	Returns an iterator pointing to the first element in the range [first,last) which compares greater than val.

***example***
```c
// lower_bound/upper_bound example
#include <iostream>     // std::cout
#include <algorithm>    // std::lower_bound, std::upper_bound, std::sort
#include <vector>       // std::vector

int main () {
  int myints[] = {10,20,30,30,20,10,10,20};
  std::vector<int> v(myints,myints+8);           // 10 20 30 30 20 10 10 20

  std::sort (v.begin(), v.end());                // 10 10 10 20 20 20 30 30

  std::vector<int>::iterator low,up;
  low=std::lower_bound (v.begin(), v.end(), 20); //          ^
  up= std::upper_bound (v.begin(), v.end(), 20); //                   ^

  std::cout << "lower_bound at position " << (low- v.begin()) << '\n';
  std::cout << "upper_bound at position " << (up - v.begin()) << '\n';

  return 0;
}
```
__结果__
==lower_bound at position 3==
==upper_bound at position 6==

3. **binary_search**
(起始地址，结束地址，要查找的数值)  返回的是是否存在这么一个数，是一个**bool值**。
(binary_search (ForwardIterator first, ForwardIterator last, const T& val))

