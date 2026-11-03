"""The gate's loop check must not depend on the script or the length.

A looping transcript is the classic STT failure on silence, so the gate
rejects one without asking the judge. Measured on UTF-8 bytes, an ordinary
Hindi, Bengali or Tamil answer looked like a loop: their letters take three
bytes that share a lead byte, and zlib collapses that. Measured over a whole
long answer, any language did, because zlib finds more shared substrings the
more text it sees.
"""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import quality

DETERMINISTIC = quality.QualityRejection(
    error_code=quality.NO_USABLE_SPEECH, reason="deterministic"
)

# One ordinary answer in seven scripts, long enough that the deterministic
# checks decide it and the judge is never asked.
ANSWERS = {
    "english": (
        "I run a small coaching centre where I teach English grammar to "
        "students from class two to class ten. I have three assistants who "
        "help me with the classes. Every month I have to collect fees from "
        "every student, keep track of who has paid, and remind the parents "
        "who are late. I also prepare weekly tests, check the answer sheets "
        "and send the marks to parents on WhatsApp. Planning the timetable "
        "for my assistants takes a lot of time too. I would like an assistant "
        "that handles the fee reminders, keeps the attendance and marks in "
        "one place, and sends the parents a short weekly report."
    ),
    "hindi": (
        "मैं एक छोटा कोचिंग सेंटर चलाता हूँ जहाँ मैं कक्षा दो से कक्षा दस तक के "
        "छात्रों को अंग्रेज़ी व्याकरण पढ़ाता हूँ। मेरे साथ तीन सहायक हैं जो कक्षाओं "
        "में मेरी मदद करते हैं। हर महीने मुझे हर छात्र से फ़ीस लेनी होती है, यह "
        "देखना होता है कि किसने भुगतान किया है, और जिन अभिभावकों की फ़ीस बाकी है "
        "उन्हें याद दिलाना होता है। मैं हर हफ़्ते टेस्ट भी बनाता हूँ, उत्तर "
        "पुस्तिकाएँ जाँचता हूँ और अंक व्हाट्सऐप पर अभिभावकों को भेजता हूँ। अपने "
        "सहायकों के लिए समय-सारणी बनाने में भी बहुत समय लगता है। मैं चाहता हूँ "
        "कि एक सहायक फ़ीस की याद दिलाने का काम संभाले, उपस्थिति और अंक एक जगह "
        "रखे, और अभिभावकों को हर हफ़्ते एक छोटी रिपोर्ट भेजे।"
    ),
    "bengali": (
        "আমি একটি ছোট কোচিং সেন্টার চালাই, যেখানে আমি দ্বিতীয় থেকে দশম শ্রেণির "
        "ছাত্রছাত্রীদের ইংরেজি ব্যাকরণ পড়াই। আমার সঙ্গে তিনজন সহকারী আছেন যাঁরা "
        "ক্লাসে আমাকে সাহায্য করেন। প্রতি মাসে আমাকে প্রত্যেক ছাত্রের কাছ থেকে "
        "বেতন নিতে হয়, কে টাকা দিয়েছে তার হিসাব রাখতে হয়, আর যাঁদের বেতন বাকি "
        "তাঁদের অভিভাবকদের মনে করিয়ে দিতে হয়। আমি প্রতি সপ্তাহে পরীক্ষাও নিই, খাতা "
        "দেখি এবং নম্বর হোয়াটসঅ্যাপে অভিভাবকদের পাঠাই। সহকারীদের জন্য রুটিন "
        "বানাতেও অনেক সময় লাগে। আমি চাই একজন সহকারী বেতনের রিমাইন্ডার পাঠাক, "
        "উপস্থিতি আর নম্বর এক জায়গায় রাখুক, আর অভিভাবকদের প্রতি সপ্তাহে একটি ছোট "
        "রিপোর্ট পাঠাক।"
    ),
    "tamil": (
        "நான் ஒரு சிறிய பயிற்சி மையம் நடத்துகிறேன், அங்கே இரண்டாம் வகுப்பு முதல் "
        "பத்தாம் வகுப்பு வரையிலான மாணவர்களுக்கு ஆங்கில இலக்கணம் கற்பிக்கிறேன். "
        "எனக்கு உதவ மூன்று உதவியாளர்கள் இருக்கிறார்கள். ஒவ்வொரு மாதமும் ஒவ்வொரு "
        "மாணவரிடமிருந்தும் கட்டணம் வசூலிக்க வேண்டும், யார் கட்டினார்கள் என்று "
        "கணக்கு வைக்க வேண்டும், கட்டணம் நிலுவையில் உள்ள பெற்றோருக்கு நினைவூட்ட "
        "வேண்டும். வாரந்தோறும் தேர்வுகள் வைத்து, விடைத்தாள்களைத் திருத்தி, "
        "மதிப்பெண்களை வாட்ஸ்அப்பில் பெற்றோருக்கு அனுப்புகிறேன். உதவியாளர்களுக்கு "
        "கால அட்டவணை தயாரிக்கவும் நிறைய நேரம் ஆகிறது. கட்டண நினைவூட்டல்களைக் "
        "கவனித்து, வருகைப் பதிவையும் மதிப்பெண்களையும் ஒரே இடத்தில் வைத்து, "
        "பெற்றோருக்கு வாராந்திர அறிக்கை அனுப்பும் ஒரு உதவியாளர் வேண்டும்."
    ),
    "arabic": (
        "أدير مركزًا تعليميًا صغيرًا أدرّس فيه قواعد اللغة الإنجليزية للطلاب من "
        "الصف الثاني إلى الصف العاشر. لديّ ثلاثة مساعدين يساعدونني في الحصص. كل "
        "شهر عليّ أن أجمع الرسوم من كل طالب، وأتابع من دفع ومن لم يدفع، وأذكّر "
        "أولياء الأمور المتأخرين. كما أُعدّ اختبارات أسبوعية، وأصحح أوراق "
        "الإجابة، وأرسل الدرجات إلى أولياء الأمور عبر واتساب. وتنظيم الجدول "
        "الزمني للمساعدين يأخذ مني وقتًا طويلًا أيضًا. أريد مساعدًا يتولى "
        "تذكيرات الرسوم، ويحفظ الحضور والدرجات في مكان واحد، ويرسل إلى أولياء "
        "الأمور تقريرًا أسبوعيًا قصيرًا."
    ),
    "russian": (
        "Я руковожу небольшим учебным центром, где преподаю английскую "
        "грамматику ученикам со второго по десятый класс. У меня три помощника, "
        "которые помогают мне на занятиях. Каждый месяц я собираю плату с "
        "каждого ученика, отслеживаю, кто заплатил, и напоминаю родителям, "
        "которые опаздывают с оплатой. Ещё я каждую неделю составляю тесты, "
        "проверяю работы и отправляю оценки родителям в WhatsApp. Составление "
        "расписания для помощников тоже отнимает много времени. Мне нужен "
        "помощник, который будет напоминать об оплате, хранить посещаемость и "
        "оценки в одном месте и каждую неделю отправлять родителям короткий отчёт."
    ),
    "chinese": (
        "我经营一家小型补习中心，给二年级到十年级的学生教英语语法。我有三位助手帮"
        "我上课。每个月我都要向每位学生收学费，记录谁已经交了，还要提醒迟交的家长。"
        "我每周还要出测验题、批改试卷，并通过微信把成绩发给家长。给助手们排课表也"
        "很花时间。我希望有一个助手能负责催缴学费，把出勤和成绩放在一个地方，并且"
        "每周给家长发一份简短的报告。"
    ),
}

# Added to the Chinese answer, this puts more than 256 distinct characters
# in the measured opening, so each one needs a two-byte code.
MORE_CHINESE = (
    "我在成都开了一家做川菜的小餐馆，店里一共有六个员工，两位厨师、三位服务员，"
    "还有一个负责收银和采购的表妹。每天早上七点我要去菜市场挑选新鲜的蔬菜和肉类，"
    "回来以后要核对昨天的账目，看看哪些菜卖得好，哪些食材剩得太多。中午和晚上是"
    "最忙的时候，外卖平台的订单经常一下子涌进来，我们得一边接单一边安排出餐顺序，"
    "有时候还会漏单或者送错地址。顾客在点评网站上留言，我也想及时回复，但是实在"
    "抽不出时间。每个月底我还要给员工算工资、交房租、跟供应商对账，这些表格都是"
    "我晚上关门以后用手机慢慢整理的，常常忙到凌晨一两点。另外我们打算明年开第二家"
    "分店，所以我还需要研究选址、租金、装修预算和招聘计划。我最希望有人帮我做的"
    "事情有三件：第一，把外卖平台和店里的订单汇总到一个地方，自动提醒厨房出餐；"
    "第二，每天晚上自动生成一份销售和库存报表，告诉我第二天该进多少货；第三，帮我"
    "整理顾客的评价，把好评和差评分类，起草回复让我确认以后再发出去。如果这些事情"
    "能自动完成，我就能把更多精力放在研发新菜和培训员工上。"
)

# What a decoder stuck on silence produces, in the same scripts.
LOOPS = {
    "english": "Thank you for watching. " * 20,
    "hindi": "धन्यवाद। " * 40,
    "bengali": "ধন্যবাদ। " * 40,
    "tamil": "நன்றி. " * 40,
    "arabic": "شكرا لكم. " * 40,
    "russian": "Спасибо за просмотр. " * 20,
    "chinese": "谢谢观看。" * 40,
}


@pytest.fixture(autouse=True)
def judge(mocker: MockerFixture) -> MagicMock:
    """Every case here must be decided without the judge."""
    return mocker.patch.object(
        quality,
        "get_openai_client",
        MagicMock(side_effect=AssertionError("the judge was asked")),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("language", sorted(ANSWERS))
async def test_an_ordinary_answer_passes_in_any_script(language: str):
    assert await quality.check_transcript_quality(ANSWERS[language]) is None


@pytest.mark.asyncio
async def test_a_long_answer_in_a_large_alphabet_passes():
    transcript = ANSWERS["chinese"] + MORE_CHINESE
    opening = transcript[: quality.REPETITION_WINDOW_CHARS]
    assert len(set(opening)) > 256
    assert await quality.check_transcript_quality(transcript) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("language", sorted(LOOPS))
async def test_a_loop_still_rejects_in_any_script(language: str):
    assert await quality.check_transcript_quality(LOOPS[language]) == DETERMINISTIC


@pytest.mark.asyncio
async def test_a_mostly_looping_opening_rejects_however_varied_the_rest():
    """More than 256 distinct characters must not lower a loop's score.

    Coding every character in two bytes once there are that many dropped
    this opening below the threshold, although most of it is a loop.
    """
    varied = "".join(chr(0x4E00 + (i * 7919) % 20000) for i in range(270))
    transcript = varied + LOOPS["chinese"] * 40
    opening = transcript[: quality.REPETITION_WINDOW_CHARS]
    assert len(set(opening)) > 256
    assert await quality.check_transcript_quality(transcript) == DETERMINISTIC


@pytest.mark.asyncio
async def test_what_follows_a_real_opening_cannot_make_it_unusable():
    """A recorder left running after the answer must not discard the answer.

    Only the opening is measured, so once a real answer fills it, the loop
    after it cannot reject the take; a long real answer would cross the
    threshold on length alone. A loop that starts inside the opening still
    counts against it.
    """
    opening = " ".join(ANSWERS[language] for language in ("english", "russian"))
    assert len(opening) > quality.REPETITION_WINDOW_CHARS
    transcript = opening + " " + LOOPS["english"] * 20
    assert await quality.check_transcript_quality(transcript) is None
