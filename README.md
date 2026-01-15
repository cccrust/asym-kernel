# asym-kernel

專門設計給 AI 寫程式的程式語言 -- 由 AI (gemini 3 pro) 指導 ccc 設計

對話連結如下：

* [ccc 與 gemini 3 pro 的對話連結](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221vt2XapnlYDgBbJe3KKY66A5V6IjKIcBr%22%5D,%22action%22:%22open%22,%22userId%22:%22111605452542833299008%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)

## 安裝環境 (install)

先準備好 rust 環境 （包含 cargo)

## 測試執行 (run)

```
(py310) cccimac@cccimacdeiMac asym-kernel % cargo run --example run_sort
warning: unused import: `DType`
 --> src/ops.rs:2:27
  |
2 | use candle_core::{Tensor, DType};
  |                           ^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: `asym-kernel` (lib) generated 1 warning (run `cargo fix --lib -p asym-kernel` to apply 1 suggestion)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.25s
     Running `target/debug/examples/run_sort`
--- A-Sym Matrix Sort Test ---
Input (Unsorted): [5.0, 1.0, 4.0, 2.0, 3.0]
Executing Op: MatrixSort
Output (Sorted):   [1.0, 2.0, 3.0, 4.0, 5.0]
Verification: PASSED
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

