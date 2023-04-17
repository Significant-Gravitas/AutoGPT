"""
reCAPTCHA[v2] challenge
TODO: VisualChallenge
"""
import os
import time
from contextlib import suppress
from typing import Optional, Union

try:
    import pydub
    from playwright.sync_api import FrameLocator, Locator, Page, expect
    from playwright.sync_api import TimeoutError
    from speech_recognition import AudioFile, Recognizer
except ImportError:
    print(
        "Playwright not installed. "
        "Please install it with 'pip install playwright pydub SpeechRecognition' to use."
    )
import requests
from urllib.request import getproxies
from .exceptions import AntiBreakOffWarning, ChallengeTimeoutException, LabelNotFoundException, RiskControlSystemArmor


class ChallengeStyle:
    AUDIO = "audio"
    VISUAL = "visual"


class ArmorUtils:
    """Various Assertion Methods for Judging Encounter reCAPTCHA"""

    @staticmethod
    def fall_in_captcha_login(page: Page) -> Optional[bool]:
        """Detect reCAPTCHA challenge encountered while logging in"""

    @staticmethod
    def fall_in_captcha_runtime(page: Page) -> Optional[bool]:
        """Detect reCAPTCHA challenge encountered at runtime"""

    @staticmethod
    def face_the_checkbox(page: Page) -> Optional[bool]:
        """face reCAPTCHA checkbox"""
        with suppress(TimeoutError):
            page.frame_locator("//iframe[@title='reCAPTCHA']")
            return True
        return False


class BaseReCaptcha:
    # <success> Challenge Passed by following the expected
    CHALLENGE_SUCCESS = "success"
    # <continue> Continue the challenge
    CHALLENGE_CONTINUE = "continue"
    # <crash> Failure of the challenge as expected
    CHALLENGE_CRASH = "crash"
    # <retry> Your proxy IP may have been flagged
    CHALLENGE_RETRY = "retry"
    # <refresh> Skip the specified label as expected
    CHALLENGE_REFRESH = "refresh"
    # <backcall> (New Challenge) Types of challenges not yet scheduled
    CHALLENGE_BACKCALL = "backcall"

    def __init__(self, dir_challenge_cache: str, style: str, debug=True, **kwargs):
        self.dir_challenge_cache = dir_challenge_cache
        self.style = style
        self.debug = debug
        self.action_name = f"{self.style.title()}Challenge"

        self.bframe = "//iframe[contains(@src,'bframe')]"
        self._response = ""

    @property
    def utils(self):
        return ArmorUtils

    @property
    def response(self):
        return self._response

    def captcha_screenshot(self, page: Union[Page, Locator], name_screenshot: str = None):
        """
        Save the challenge screenshot, which needs to be done after get_label
        :param page:
        :param name_screenshot: filename of the Challenge image
        :return:
        """
        if hasattr(self, "label_alias") and hasattr(self, "label"):
            _suffix = self.label_alias.get(self.label, self.label)
        else:
            _suffix = self.action_name
        _filename = (
            f"{int(time.time())}.{_suffix}.png" if name_screenshot is None else name_screenshot
        )
        _out_dir = os.path.join(os.path.dirname(self.dir_challenge_cache), "captcha_screenshot")
        _out_path = os.path.join(_out_dir, _filename)
        os.makedirs(_out_dir, exist_ok=True)

        # FullWindow screenshot or FocusElement screenshot
        page.screenshot(path=_out_path)
        return _out_path

    @staticmethod
    def _activate_recaptcha(page: Page):
        """Process checkbox activation reCAPTCHA"""
        # --> reCAPTCHA iframe
        activator = page.frame_locator("//iframe[@title='reCAPTCHA']").locator(
            ".recaptcha-checkbox-border"
        )
        activator.click()
        # self.log("Active reCAPTCHA")

        # Check reCAPTCHA accessible status for the checkbox-result
        with suppress(TimeoutError):
            if status := page.locator("#recaptcha-accessible-status").text_content(timeout=2000):
                raise AntiBreakOffWarning(status)

    def _switch_to_style(self, page: Page) -> Optional[bool]:
        """
        Toggle authentication mode used before anti_checkbox() execution
        :param page:
        :raise AntiBreakOffWarning: Unable to switch to <voiceprint verification mode>
        :return:
        """
        frame_locator = page.frame_locator(self.bframe)
        # Switch to <voiceprint verification mode> or stay in <visual verification mode>
        if self.style == ChallengeStyle.AUDIO:
            switcher = frame_locator.locator("#recaptcha-audio-button")
            expect(switcher).to_be_visible()
            switcher.click()
        # self.log("Accept the challenge", style=self.style)
        return True

    def anti_recaptcha(self, page: Page):
        """Execution flow for human-machine challenges"""
        # [⚔] Activate the reCAPTCHA and switch to
        # <voiceprint verification mode> or <visual verification mode>
        try:
            self._activate_recaptcha(page)
        except AntiBreakOffWarning as _unused_err:
            # logger.info(_unused_err)
            return
        return self._switch_to_style(page)


class AudioChallenger(BaseReCaptcha):
    def __init__(self, dir_challenge_cache: str, debug: Optional[bool] = True, **kwargs):
        super().__init__(
            dir_challenge_cache=dir_challenge_cache,
            style=ChallengeStyle.AUDIO,
            debug=debug,
            kwargs=kwargs,
        )

    @staticmethod
    def _get_audio_download_link(fl: FrameLocator) -> Optional[str]:
        """Returns the download address of the sound source file."""
        for _ in range(5):
            with suppress(TimeoutError):
                # self.log("Play challenge audio")
                fl.locator("//button[@aria-labelledby]").click(timeout=1000)
                break
            with suppress(TimeoutError):
                header_text = fl.locator(".rc-doscaptcha-header-text").text_content(timeout=1000)
                if "Try again later" in header_text:
                    raise ConnectionError(
                        "Your computer or network may be sending automated queries."
                    )

        # Locate the sound source file url
        try:
            audio_url = fl.locator("#audio-source").get_attribute("src")
        except TimeoutError:
            raise RiskControlSystemArmor("Trapped in an inescapable risk control context")
        return audio_url

    @staticmethod
    def _handle_audio(dir_challenge_cache: str, audio_url: str) -> str:
        """
        Location, download and transcoding of audio files
        Args:
            dir_challenge_cache:
            audio_url: reCAPTCHA Audio Link address

        Returns:

        """
        # Splice audio cache file path
        timestamp_ = int(time.time())
        path_audio_mp3 = os.path.join(dir_challenge_cache, f"audio_{timestamp_}.mp3")
        path_audio_wav = os.path.join(dir_challenge_cache, f"audio_{timestamp_}.wav")

        # Download the sound source file to the local
        # self.log("Downloading challenge audio")
        _request_asset(audio_url, path_audio_mp3)

        # Convert audio format mp3 --> wav
        # self.log("Audio transcoding MP3 --> WAV")
        pydub.AudioSegment.from_mp3(path_audio_mp3).export(path_audio_wav, format="wav")
        # self.log("Transcoding complete", path_audio_wav=path_audio_wav)

        # Returns audio files in wav format to increase recognition accuracy
        return path_audio_wav

    @staticmethod
    def _parse_audio_to_text(path_audio_wav: str) -> str:
        """
        Speech recognition, audio to text
        :param path_audio_wav: reCAPTCHA Audio The local path of the audio file（.wav）
        :exception speech_recognition.RequestError: Need to suspend proxy
        :exception http.client.IncompleteRead: Poor Internet Speed，
        :return:
        """
        # Internationalized language format of audio files, default en-US American pronunciation.
        language = "en-US"

        # Read audio into and cut into a frame matrix
        recognizer = Recognizer()
        audio_file = AudioFile(path_audio_wav)
        with audio_file as stream:
            audio = recognizer.record(stream)

        # Returns the text corresponding to the short audio(str)，
        # en-US Several words that are not sentence patterns
        # self.log("Parsing audio file ... ")
        audio_answer = recognizer.recognize_google(audio, language=language)
        # self.log("Analysis completed", audio_answer=audio_answer)

        return audio_answer

    @staticmethod
    def _submit_text(fl: FrameLocator, text: str) -> Optional[bool]:
        """
        Submit reCAPTCHA man-machine verification
        The answer text information needs to be passed in,
        and the action needs to stay in the submittable frame-page.
        :param fl:
        :param text:
        :return:
        """
        with suppress(NameError, TimeoutError):
            input_field = fl.locator("#audio-response")
            input_field.fill("")
            input_field.fill(text.lower())
            # self.log("Submit the challenge")
            input_field.press("Enter")
            return True
        return False

    @staticmethod
    def is_correct(page: Page) -> Optional[bool]:
        """Check if the challenge passes"""
        with suppress(TimeoutError):
            err_resp = page.locator(".rc-audiochallenge-error-message")
            if _unused_msg := err_resp.text_content(timeout=2000):
                return False
        return True

    def anti_recaptcha(self, page: Page):
        if super().anti_recaptcha(page) is not True:
            return

        # [⚔] Register Challenge Framework
        frame_locator = page.frame_locator(self.bframe)
        # [⚔] Get the audio file download link
        audio_url: str = self._get_audio_download_link(frame_locator)
        # [⚔] Audio transcoding（MP3 --> WAV）increase recognition accuracy
        path_audio_wav: str = self._handle_audio(
            dir_challenge_cache=self.dir_challenge_cache, audio_url=audio_url
        )
        # [⚔] Speech to text
        audio_answer: str = self._parse_audio_to_text(path_audio_wav)
        # [⚔] Locate the input box and fill in the text
        if self._submit_text(frame_locator, text=audio_answer) is not True:
            # self.log("reCAPTCHA Challenge submission failed")
            raise ChallengeTimeoutException
        # Judging whether the challenge is successful or not
        # Get response of the reCAPTCHA
        if self.is_correct(page):
            self._response = page.evaluate("grecaptcha.getResponse()")
            return self.CHALLENGE_SUCCESS
        return self.CHALLENGE_RETRY


class _VisualChallenger(BaseReCaptcha):
    TASK_OBJECT_DETECTION = "ObjectDetection"
    TASK_BINARY_CLASSIFICATION = "BinaryClassification"

    FEATURE_DYNAMIC = "rc-imageselect-dynamic-selected"
    FEATURE_SELECTED = "rc-imageselect-tileselected"

    # TODO
    # crosswalks
    # stairs
    # vehicles
    # tractors
    # taxis
    # chimneys
    # mountains or hills
    # bridge
    # cars
    label_alias = {
        "zh": {
            "消防栓": "fire hydrant",
            "交通灯": "traffic light",
            "汽车": "car",
            "自行车": "bicycle",
            "摩托车": "motorcycle",
            "公交车": "bus",
            "船": "boat",
        },
        "en": {
            "a fire hydrant": "fire hydrant",
            "traffic lights": "traffic light",
            "car": "car",
            "bicycles": "bicycle",
            "motorcycles": "motorcycle",
            "bus": "bus",
            "buses": "bus",
            "cars": "car",
            "boats": "boat",
        },
    }

    def __init__(
            self,
            dir_challenge_cache: str,
            dir_model: str,
            onnx_prefix: Optional[str] = None,
            screenshot: Optional[bool] = False,
            debug: Optional[bool] = True,
            **kwargs,
    ):
        super().__init__(
            dir_challenge_cache=dir_challenge_cache,
            style=ChallengeStyle.VISUAL,
            debug=debug,
            kwargs=kwargs,
        )
        self.dir_model = dir_model
        self.onnx_prefix = onnx_prefix
        self.screenshot = screenshot
        self.prompt: str = ""
        self.label: str = ""
        self.lang: str = "en"
        self.label_alias = _VisualChallenger.label_alias[self.lang]

        # _oncall_task "object-detection" | "binary-classification"
        self._oncall_task = None

    def reload(self, page: Page):
        """Overload Visual Challenge :: In the BFrame"""
        page.frame_locator(self.bframe).locator("#recaptcha-reload-button").click()
        page.wait_for_timeout(1000)

    def check_oncall_task(self, page: Page) -> Optional[str]:
        """Identify the type of task：Detection task or classification task"""
        # Usually, when the number of clickable pictures is 16, it is an object detection task,
        # and when the number of clickable pictures is 9, it is a classification task.
        image_objs = page.frame_locator(self.bframe).locator("//td[@aria-label]")
        self._oncall_task = (
            self.TASK_OBJECT_DETECTION
            if image_objs.count() > 9
            else self.TASK_BINARY_CLASSIFICATION
        )
        return self._oncall_task

    def get_label(self, page: Page):
        def split_prompt_message(prompt_message: str) -> str:
            prompt_message = prompt_message.strip()
            return prompt_message

        # Captcha prompts
        label_obj = page.frame_locator(self.bframe).locator("//strong")
        self.prompt = label_obj.text_content()
        # Parse prompts to model label
        try:
            _label = split_prompt_message(prompt_message=self.prompt)
        except (AttributeError, IndexError):
            raise LabelNotFoundException("Get the exception label object")
        else:
            self.label = _label
            # self.log(
            #     message="Get label", label=f"「{self.label}」", task=f"{self.check_oncall_task(page)}"
            # )

    def select_model(self):
        """Optimizing solutions based on different challenge labels"""
        # label_alias = self.label_alias.get(self.label)
        return self.yolo_model

    def mark_samples(self, page: Page):
        """Get the download link and locator of each challenge image"""
        samples = page.frame_locator(self.bframe).locator("//td[@aria-label]")
        for index in range(samples.count()):
            fn = f"{int(time.time())}_/Challenge Image {index + 1}.png"
            self.captcha_screenshot(samples.nth(index), name_screenshot=fn)
            # self.log("save image", fn=fn)
        image_link = (
            page.frame_locator(self.bframe)
            .locator("//td[@aria-label]//img")
            .first.get_attribute("src")
        )
        # self.log(image_link)

    def check_positive_element(
            self, sample: Locator, model, screenshot: Optional[bool] = False
    ) -> Optional[bool]:
        """Review positive samples"""
        result = model.solution(img_stream=sample.screenshot(), label=self.label_alias[self.label])

        # Pass: Hit at least one object
        if result:
            sample.click()

        # Check result of the challenge.
        if screenshot or self.screenshot:
            _filename = f"{int(time.time())}.{model.flag}.{self.label_alias[self.label]}.png"
            self.captcha_screenshot(sample, name_screenshot=_filename)

        return result

    def challenge(self, page: Page, model):
        """Image classification, element clicks, answer submissions"""

        def hit_dynamic_samples(target: list):
            if not target:
                return
            for i in target:
                locator_ = f'//td[@tabindex="{i + 4}"]'
                # Gradient control
                # Ensure that the pictures fed into the model are correctly exposed.
                with suppress(TimeoutError, AssertionError):
                    expect(page.frame_locator(self.bframe).locator(locator_)).to_have_attribute(
                        "class", self.FEATURE_DYNAMIC
                    )
                dynamic_element = page.frame_locator(self.bframe).locator(locator_)
                result_ = self.check_positive_element(sample=dynamic_element, model=model)
                if not result_:
                    target.remove(i)
            return hit_dynamic_samples(target)

        is_dynamic = None
        dynamic_index = []
        samples = page.frame_locator(self.bframe).locator("//td[@aria-label]")
        for index in range(samples.count()):
            result = self.check_positive_element(sample=samples.nth(index), model=model)
            if is_dynamic is None:
                motion_status = (
                    page.frame_locator(self.bframe)
                    .locator(f'//td[@tabindex="{index + 4}"]')
                    .get_attribute("class")
                )
                if self.FEATURE_SELECTED in motion_status:
                    is_dynamic = False
                elif self.FEATURE_DYNAMIC in motion_status:
                    is_dynamic = True
            if result:
                dynamic_index.append(index)

        # Winter is coming
        if is_dynamic:
            hit_dynamic_samples(target=dynamic_index)
        # Submit challenge
        page.frame_locator(self.bframe).locator("//button[@id='recaptcha-verify-button']").click()

    def check_accessible_status(self, page: Page) -> Optional[str]:
        """Judging whether the challenge was successful"""
        try:
            prompt_obj = page.frame_locator(self.bframe).locator(
                "//div[@class='rc-imageselect-error-select-more']"
            )
            prompt_obj.wait_for(timeout=1000)
        except TimeoutError:
            try:
                prompt_obj = page.frame_locator(self.bframe).locator(
                    "rc-imageselect-incorrect-response"
                )
                prompt_obj.wait_for(timeout=1000)
            except TimeoutError:
                return self.CHALLENGE_SUCCESS

        prompts = prompt_obj.text_content()
        return prompts

    def tactical_retreat(self, page: Page) -> Optional[str]:
        """
        「blacklist mode」 skip unchoreographed challenges
        :param page:
        :return: the screenshot storage path
        """
        if self.label_alias.get(self.label):
            return self.CHALLENGE_CONTINUE

        # Save a screenshot of the challenge
        with suppress(TimeoutError):
            challenge_container = page.frame_locator(self.bframe).locator(
                "//body[@class='no-selection']"
            )
            path_screenshot = self.captcha_screenshot(challenge_container)
            # logger.warning(
            #     "Types of challenges not yet scheduled - "
            #     f"label=「{self.label}」 prompt=「{self.prompt}」 "
            #     f"{path_screenshot=} {page.url=}"
            # )

        return self.CHALLENGE_BACKCALL

    def anti_recaptcha(self, page: Page):
        """

        >> NOTE:

        ——————————————————————————————————————————————————————————————————————————

        In the accessible `detection task`, reCAPTCHA v2 implements a "multi-group multi-unit target"
        challenge presentation scheme, that is:

            1. The target to be recognized by 'NEXT' appears normally,
            usually with multiple unit targets.

            2. In the scene information composed of 16 images in 'SKIP',
            there is no actual appearance of the target to be recognized,
            and the challenge needs to be skipped at this time.

            3. Generally, encountering a detection task means you have been marked
            as a high-risk visitor and will frequently encounter labels
            such as traffic_lights and crosswalks.

        In best practices, it is recommended to switch to classification tasks through `reload`.

        ——————————————————————————————————————————————————————————————————————————

        In the accessible `classification task`, reCAPTCHA v2 implements a
        "single group gradient noise addition" challenge presentation scheme, that is:

            1. Unlike detection tasks in 'VERIFY', classification tasks usually
            only have one set of images that need classification; click verify to submit challenges.

            2. Usually, there are more than nine pictures in one set of
            classification tasks (but only nine are presented at once).

            - (OptionalStyle) After clicking on an image, the clicked
            grid handle will replace a new image **gradually**,

        The higher your threat score or request frequency within
        a short period of time, the longer it takes for gradients.

            - When all positive objects in all grids disappear—that is
            when all positives currently viewed have been selected,

            Clicking submit can pass through challenges (requiring an accuracy rate of 100%)
            otherwise error-prompt will occur.

            - When encountering error-prompt situations ,the challenge does not refresh
            automatically but stays in its current context; you need to continue selecting all positive images.

        3. "Human-visible" noise images will appear in classification tasks.

        ——————————————————————————————————————————————————————————————————————————

        First-order state machine:

            - Reload challenge: After refreshing the challenge, both labels and images change together.

            - rc-imageselect-dynamic-selected: Additional CLASS_NAME for grid buttons during gradient transitions.

            - rc-imageselect-tileselected: Normal selection effect.

        Exception status table:

            - Please select all matching images. There are still uneliminated positive samples in the current context.

            - Please also check the new images. In classification tasks, submit challenges while waiting for image grids to gradually load completely.

            When this prompt pops up, the challenge context remains unchanged—that is, prompts and images do not change.

            - Please try again. The score is too low; please retry.

        """
        if super().anti_recaptcha(page) is not True:
            return
        # [⚔] Register Challenge Framework
        # TODO: TASK_OBJECT_DETECTION, more label
        for _ in range(3):
            # [⚔] Skip objects detection tasks and unprepared classification tasks
            for _ in range(10):
                # [⚔] Get challenge labels
                self.get_label(page)
                if self._oncall_task == self.TASK_OBJECT_DETECTION:
                    self.reload(page)
                elif self.tactical_retreat(page) == self.CHALLENGE_BACKCALL:
                    self.reload(page)
                else:
                    break

            model = self.select_model()
            self.challenge(page, model=model)
            self.captcha_screenshot(page)
            if drop := self.check_accessible_status(page) == self.CHALLENGE_SUCCESS:
                self._response = page.evaluate("grecaptcha.getResponse()")
                return drop
        else:
            input("This method has not been implemented yet, press any key to exit the program.")


def _request_asset(asset_download_url: str, asset_path: str):
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.27"
    }

    # FIXME: PTC-W6004
    #  Audit required: External control of file name or path
    with open(asset_path, "wb") as file, requests.get(
            asset_download_url, headers=headers, stream=True, proxies=getproxies()
    ) as response:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
