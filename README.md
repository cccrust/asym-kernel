# asym-kernel

專門設計給 AI 寫程式的程式語言 -- 由 AI (gemini 3 pro) 指導 ccc 設計

對話連結如下：

* [ccc 與 gemini 3 pro 的對話連結](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221vt2XapnlYDgBbJe3KKY66A5V6IjKIcBr%22%5D,%22action%22:%22open%22,%22userId%22:%22111605452542833299008%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)

## 安裝環境 (install)

先準備好 rust 環境 （包含 cargo)

## 測試執行 (run)

測試： [run_sort.rs](examples/run_sort.rs)

```
(py310) cccimac@cccimacdeiMac asym-kernel % cargo run --example run_sort
   Compiling asym-kernel v0.1.0 (/Users/cccimac/Desktop/ccc/project/asym-kernel)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.56s
     Running `target/debug/examples/run_sort`
--- A-Sym Matrix Sort Test ---
Input (Unsorted): [5.0, 1.0, 4.0, 2.0, 3.0]
    Executing: MatrixSort
Output (Sorted):   [1.0, 2.0, 3.0, 4.0, 5.0]
Verification: PASSED
```

測試： [run_branch.rs](examples/run_branch.rs)

```
(py310) cccimac@cccimacdeiMac asym-kernel % cargo run --example run_branch
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.22s
     Running `target/debug/examples/run_branch`
--- Scenario 1: High Mean Data (100, 200, 300) ---
--> [Decision] Current Mean: 200.0000, Threshold: 50
    Path: TRUE_PATH (Mean > Threshold)
    Executing: Normalize
Final High Result: [-1.2247449, 0.0, 1.2247449]

--- Scenario 2: Low Mean Data (1, 3, 2) ---
--> [Decision] Current Mean: 2.0000, Threshold: 50
    Path: FALSE_PATH (Mean <= Threshold)
    Executing: Square
    Executing: MatrixSort
Final Low Result: [1.0, 4.0, 9.0]
```

測試： [run_memory.rs](./examples/run_memory.rs)

```
(py310) cccimac@cccimacdeiMac asym-kernel % cargo run --example run_memory
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.25s
     Running `target/debug/examples/run_memory`
--- Task A: Process and Store ---
    Executing: Square
    --> [Memory] Stored tensor to slot: 'feature_01'

--- Task B: Load and Finalize ---
    --> [Memory] Loaded tensor from slot: 'feature_01'
    Executing: Normalize
Final Result from Memory: [-1.1111674, -0.20203052, 1.3131976]
```

## 人類評論

我想，或許很快，AI 會設計出

全世界最多『人+AI』使用的『程式語言』

到那個時候，身為人類程序員

就差不多要退出世界舞台了 ...

但是，你可以叫 AI 用那個 AI 發明的程式語言，寫出你要的程式

於是，我叫 Gemini Pro 設計一個這樣的語言 （稱為 A-Sym)

他設計出來了，我看不太懂 ....

我照著做，第一版他說用 rust 寫

AI 除錯幾次後，程式還真的能跑，目前最大的瓶頸是我的理解力 ....

