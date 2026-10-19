# Tổng hợp kết quả DeepTherm Learning Curve & Baseline None

Tất cả kết quả dưới đây là MAE của ΔH298 trên tập Test, đơn vị là **kcal/mol**.

## 1. Learning Curve (Có Prior)
| Tỷ lệ Dữ liệu | Random 0 | Random 1 | Random 2 | Trung bình |
|---|---|---|---|---|
| 10% | 3.007 | 3.085 | 3.158 | **3.083** |
| 20% | 2.261 | 2.361 | 2.381 | **2.334** |
| 40% | 1.777 | 1.879 | 1.866 | **1.841** |
| 60% | 1.700 | 1.624 | 1.851 | **1.725** |

## 2. Learning Curve (Không Prior - Baseline None)
| Tỷ lệ Dữ liệu | Random 0 | Random 1 | Random 2 | Trung bình |
|---|---|---|---|---|
| 10% | 5.160 | 4.933 | 4.787 | **4.960** |
| 90% | 2.033 | (chưa chạy) | (chưa chạy) | **2.033** |

## 3. DeepTherm Complexity Split (90% Data)
| Cấu hình | Complexity 0 |
|---|---|
| Không Prior | **4.206** |
