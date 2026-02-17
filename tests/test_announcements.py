import io
import json
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bagofwords import BagOfWord

@pytest.mark.parametrize("document, expected", [
(
        """
        Đăng ký học phần
        """,
        True
    ),
    (
        """
        Tiêu đề là ĐKHP!
        """,
        True
    ),
    (
        """
        Thông báo khảo sát ý kiến SV về hoạt động giảng dạy của GV - NH 2025-2026
        """,
        False
    ),
    (
        """
        Danh sách chính thức lớp Huredee 7 học tiếng Nhật miễn phí và tư vấn việc làm tại Nhật Bản do tổ chức Huredee tài trợ
        """,
        False
    ),
    (
        """
        Nộp hồ sơ du học Viện Công nghệ IIST, Đại học Hosei
        """,
        False
    ),
    (
        """
        Thông báo Lịch học ôn tập Olympic Toán năm học 2025-2026
        """,
        False
    ),
    (
        """
        Thông báo nhận đơn nhập học lại, bảo lưu, chuyển ngành, song ngành, thôi học HK 2 2025-2026
        """,
        False
    ),
    (
        """
        Thông báo về việc nhận bằng tốt nghiệp đợt 4 năm 2025
        """,
        False
    ),
    (
        """
        Thông báo lịch thi cuối kỳ học kỳ 1 năm học 2025-2026. 
        Thân chào các bạn sinh viên;

        Phòng Đào tạo Đại học thông báo lịch thi Cuối kỳ học kỳ 1 năm học 2025-2026 chi tiết trong file đính kèm.

        Nếu các bạn có thắc mắc vui lòng liên hệ P.ĐTĐH/VPCCTĐB để được hỗ trợ.

        Ghi chú: các bạn hoãn thi đăng ký thi lại đến hết ngày 14/12/2025

        Trân trọng

        Phòng Đào tạo Đại học.
        """,
        False
    ),
    (
        """
        Thông báo thu học phí học kỳ 1, năm học 2025-2026 trình độ ĐTĐH CT liên kết BCU, VB2CQ, LTĐH, Song ngành
        Chào các bạn sinh viên!

        Phòng KHTC có đăng tải Thông báo thu học phí học kỳ 1, năm học 2025-2026 trình độ ĐTĐH Chương trình liên kết BCU, VB2CQ, LTĐH, Song ngành lên website phòng KHTC TẠI ĐÂY

        Các bạn vào xem thông tin nhé.

        Trân trọng.

        Phòng Đào tạo Đại học.
        """,
        False
    ),
    (
        """
        Thông báo nghỉ giảng dạy để tham dự Lễ kỷ niệm 43 năm Ngày Nhà giáo Việt Nam 20/11/2025

        Kính chào Quý Thầy, Cô và các bạn sinh viên!

        Trường Đại học CNTT có kế hoạch tổ chức Lễ Kỷ niệm 43 năm Ngày Nhà giáo Việt Nam 20/11 do vậy các lớp học vào buổi sáng ngày 19/11/2025 được nghỉ và Thầy, Cô đăng ký dạy bù vào tuần dự trữ trong khoảng thời gian từ ngày 29/12/2025 đến ngày 03/01/2026 (không tính ngày nghỉ lễ Tết Dương lịch) theo kế hoạch năm học.

        ---

        TBN
        """,
        False
    ),
    (
        """
        Thông báo lịch thi cuối kỳ các môn Anh văn học kỳ 1 năm học 2025-2026
        Thân chào các bạn sinh viên,

        Phòng Đào tạo Đại học thông báo lịch thi Cuối kỳ các môn Anh văn học kỳ 1 năm học 2025-2026 (chi tiết trong file đính kèm).

        Lưu ý:

        - Về lịch thi Vấn đáp sinh viên theo dõi thông tin theo file đính kèm của thông báo.

        - Về lịch thi Viết (Phòng máy) sinh viên theo dõi thông tin lịch thi tại cổng thông tin sinh viên: https://student.uit.edu.vn.

            Trân trọng

        Phòng Đào tạo Đại học.
        """,
        False
    ),
    (
        """
        Kết quả ĐKHP (đợt 2) học kỳ 1 năm học 2025-2026_Chương trình chuẩn.
        Chào các bạn sinh viên!

        Phòng ĐTĐH thông báo kết quả ĐKHP (đợt 2) học kỳ 1 năm học 2025-2026 của các lớp chương trình chuẩn tại đây.

        Trân trọng.

        Cv. LTT Phương
        """,
        True
    )
])
def test_announcements(document: str, expected: bool):
    with io.open(ROOT_DIR / "training_data" / "announcements.json", 'r', encoding="utf-8") as file:
        data_sets = json.load(file)["data"]
    
    try:
        training_sets = [("{0}. {1}".format(data_set["headline"], data_set["document"]), data_set["label"]) for data_set in data_sets]
    except KeyError as e:
        print(f"Training data is deformed, make sure to include a {e} key in the set")
        return
    # for training_set in training_sets:
    #     print(f"document: {training_set[0]} | label: {training_set[1]}")
    results = BagOfWord.predict_from_training_set(training_sets=training_sets, test_documents=[document])
    assert results[0] == expected