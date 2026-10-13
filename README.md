# Spark

Mã độc lập để pretrain draft và train policy bằng GRPO, với hai lựa chọn:

- `tree`: draft một lớp dự đoán trạng thái ẩn và phân phối token. Mở rộng cây theo xác suất tích lũy, giữ các nút có độ tin cậy cao, xác minh bằng tree attention và đi theo token do target lấy mẫu. Loss draft gồm Smooth L1 trạng thái ẩn và soft cross entropy.
- `block`: draft ba lớp dự đoán song song các vị trí MASK. Mỗi block chỉ nhìn thấy lịch sử trước anchor và các vị trí trong chính block đó. Chia sẻ embedding/head của target; dùng soft cross entropy với trọng số giảm theo vị trí. Proposal greedy được xác minh bằng rejection sampling, có token phục hồi hoặc bonus của target.

Thêm `--correction` khi train để bật hiệu chỉnh online: projector được học qua các batch; ma trận hiệu chỉnh được đặt lại ở đầu mỗi batch rollout và cập nhật từ các context thực sự được xác minh. Cập nhật chỉ dùng hợp top-k của hai phân phối. Mặc định tắt.

Không có logging, thanh tiến trình, dịch vụ nền hoặc phụ thuộc vào các project khác trong workspace. Các thư viện vẫn có thể báo cảnh báo/lỗi. Checkpoint là đầu ra duy nhất của chương trình.

## Cấu trúc mã nguồn

```text
Spark/
├── pretrain.py              # Vòng lặp pretrain draft
├── train.py                 # Vòng lặp GRPO và train draft online
├── helper/
│   ├── __init__.py
│   ├── arguments.py         # Tham số dòng lệnh và kiểm tra cấu hình
│   ├── data.py              # Đọc dữ liệu, chat template và chia batch
│   ├── runtime.py           # Nạp target, precision và lấy context features
│   ├── objectives.py        # Loss policy/draft và cập nhật draft
│   ├── optimizers.py        # FP32 master weights và tích lũy gradient
│   ├── generation.py        # Sinh và xác minh token cho tree/block
│   ├── correction.py        # Hiệu chỉnh online tùy chọn
│   ├── rewards.py           # Reward đáp án toán và định dạng
│   └── checkpoints.py       # Lưu/nạp checkpoint và RNG
├── models/
│   ├── __init__.py
│   ├── draft.py             # Kiến trúc draft tree/block
│   └── policy.py            # Adapter policy và khôi phục trọng số
├── requirements.txt
└── README.md
```

`pretrain.py` và `train.py` giữ vai trò điều phối. Các module trong `helper/` và `models/` chỉ chứa phần triển khai cần cho hai entry point này. Lệnh chạy và định dạng checkpoint giữ nguyên.

## Môi trường và đầu vào

Dùng Python 3.10 trở lên và bản PyTorch có CUDA phù hợp với GPU. Nếu môi trường đã có các dependency trong `requirements.txt`, dùng trực tiếp môi trường đó. Nếu chưa có:

```bash
python -m pip install -r requirements.txt
```

Mỗi lệnh chạy một process trên một GPU. Chọn GPU bằng `CUDA_VISIBLE_DEVICES`. Không dùng `torchrun`. Code dùng PyTorch SDPA; không cần biên dịch extension CUDA. Model/tokenizer phải có sẵn ở máy; chương trình không tải qua mạng.

Target cần decoder dense, RMSNorm, gated SiLU, grouped-query attention và RoPE chuẩn. Các checkpoint dùng attention dạng sliding window, MoE hoặc RoPE có scaling không thuộc phạm vi hỗ trợ. Tokenizer cần chat template giữ nguyên token prefix khi thêm lượt hội thoại và có EOS. Với `block`, embedding phải có một hàng không thuộc vocabulary của tokenizer để làm MASK; chương trình tự tìm hàng đó, hoặc nhận `--mask-id`.

Đặt ba biến môi trường bằng các giá trị của bạn trước khi chạy:

- `MODEL`: thư mục target và tokenizer.
- `PRETRAIN_DATA`: file hội thoại.
- `TRAIN_DATA`: file câu hỏi và đáp án.

Dữ liệu có thể là JSON (mảng object), JSONL hoặc Parquet. Schema pretrain:

```json
[
  {"messages": [
    {"role": "user", "content": "1 + 1 bằng bao nhiêu?"},
    {"role": "assistant", "content": "2"}
  ]}
]
```

Cũng nhận `conversations` với `from`/`value`. Chỉ các lượt assistant đóng góp vào loss; các token bị cắt ở `--max-length` không được dùng.

Schema train:

```json
[
  {"question": "1 + 1 bằng bao nhiêu?", "answer": "2"}
]
```

Có thể thay `question` bằng `prompt` dạng chuỗi hoặc danh sách message, hoặc dùng `messages`; thay `answer` bằng `reward_model.ground_truth`. `question` được bọc bằng chỉ dẫn suy luận và trả lời toán; `prompt`/`messages` dùng nội dung bạn cung cấp. Prompt dài hơn giới hạn bị từ chối. Reward là độ đúng của đáp án toán cộng `0.2` nếu có đúng một đoạn `\n</think>\n`. Ground truth không parse được sẽ báo lỗi trước khi train.

## Pretrain

Chạy từ thư mục `Spark`:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
  --method tree --model "$MODEL" --data "$PRETRAIN_DATA" \
  --output checkpoints/tree_pretrain

CUDA_VISIBLE_DEVICES=1 python pretrain.py \
  --method block --model "$MODEL" --data "$PRETRAIN_DATA" \
  --output checkpoints/block_pretrain
```

Mặc định: target đóng băng, 5 epoch, learning rate `5e-5`, batch size `8`, accumulation `4`, giới hạn `2048` token. Với `block`: block size `16`, `16` anchor mỗi chuỗi. Có thể ghi đè bằng `--layers`, `--block-size`, `--anchors`, `--max-length`, `--lr`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --method tree --model "$MODEL" --data "$TRAIN_DATA" \
  --pretrained checkpoints/tree_pretrain/latest.pt \
  --output checkpoints/tree_train

CUDA_VISIBLE_DEVICES=1 python train.py \
  --method block --model "$MODEL" --data "$TRAIN_DATA" \
  --pretrained checkpoints/block_pretrain/latest.pt \
  --output checkpoints/block_train
```

Để bật hiệu chỉnh, thêm `--correction` và dùng một `--output` riêng. Hai checkpoint pretrain được dùng chung cho các lần train có hoặc không có hiệu chỉnh tương ứng.

Mặc định: 1 epoch, `8` prompt mỗi batch, `8` response mỗi prompt, accumulation `4`, target learning rate `1e-6`, draft learning rate `1e-4`, temperature `1.0`, top-p `0.95`, KL coefficient `0.04`. Giới hạn prompt và response đều `2048` token. GRPO chuẩn hóa reward trong từng nhóm response, lấy trung bình loss theo chuỗi; các nhóm có reward đồng nhất không cập nhật target nhưng vẫn train draft.

`--policy-update native` mặc định dùng adapter hạng `64`, hệ số `32` cho `tree`, và cập nhật toàn bộ target cho `block`. Có thể chọn rõ `--policy-update adapter` hoặc `--policy-update full` để dùng cùng phạm vi cập nhật ở hai nhánh. Full training giữ một reference model đóng băng; cần đủ VRAM cho model, reference, FP32 master weights và optimizer. Có thể chuyển reference sang CPU bằng `--reference-device cpu`, đổi lại tốc độ thấp hơn.

## Iteration và checkpoint

Accumulation là **tích lũy gradient thật**. Một iteration gồm tối đa 4 batch; pretrain tương ứng 32 hội thoại, train tương ứng 32 prompt / 256 response. Target cập nhật tối đa một lần mỗi iteration; draft cập nhật một lần mỗi batch rollout. Forward/backward được xử lý lần lượt theo chuỗi để giảm bộ nhớ, nên `--batch-size` là kích thước batch logic. Batch/iteration cuối có thể ngắn hơn.

Mặc định lưu mỗi `50` iteration và khi kết thúc; đổi bằng `--save-every`. File `step_N.pt` chứa draft, policy nếu đang train, optimizer, vị trí dữ liệu và RNG; `latest.pt` trỏ tới checkpoint mới nhất. Không lưu đường dẫn đầu vào trong checkpoint. Hai entry point chỉ đọc checkpoint do chính code này tạo, không đọc trực tiếp định dạng của backend khác.

Để chạy tiếp, dùng lại lệnh tương ứng cùng các tham số và thêm:

```bash
--resume checkpoints/tree_train/latest.pt
```

Với pretrain, trỏ tới checkpoint pretrain. Không cần `--pretrained` khi resume train. Giữ nguyên target/tokenizer, dữ liệu, batch size, accumulation, số epoch và các tham số thuật toán; chương trình kiểm tra cấu hình và checksum dữ liệu. `--max-steps N` cho phép dừng sau iteration N và lưu để chạy tiếp.

Đây là bản triển khai độc lập của thuật toán. Batching, thứ tự lấy mẫu và tích lũy gradient được quy định ở trên; thời gian chạy và trajectory ngẫu nhiên không được coi là tương đương backend khác. Đã kiểm tra bằng target nhỏ trên PyTorch 2.4.1 và Transformers 4.57.1; cần kiểm tra cấu hình model thực tế trước một đợt train dài.
