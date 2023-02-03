"""
Alot of hard coding here
"""

def hightlightAITextHTML(texts, scores, labels):
    """
    Return hightlighted AI text if label == 1
    """
    def rowHTML(text, header=False):
        if header:
            return f'<th style="padding-bottom: 0.5em">{text}</th>'
        else:
            return f'<td style="padding-bottom: 0.5em">{text}</td>'

    def HTML():
        results = f'<table><tr>{rowHTML("Score", True)}<colspan>{rowHTML("Text [The highlighted green text are most probably AI generated]", True)}</tr>'
        for i in range(len(texts)):
            if labels[i] == 1:
                results += '<tr style="background:#adff2f">'
            else:
                results += '<tr>'

            results += rowHTML(scores[i]) + rowHTML(texts[i])
            results += "</tr>"
        results  += "</table>"
        return results
    return HTML()

def noticeBoardMarkDown():
    return """
    # NoticeBoard

    <h4>Release Version 1.1</h4>
    We have implemented DetectGPT, a method that detects samples from pre-trained LLMs using the local curvature of the model's log probability function. (Based on the research paper published by Stanford University, <a href="https://arxiv.org/pdf/2301.11305.pdf">https://arxiv.org/pdf/2301.11305.pdf</a>) <br>
    <sub>3-Feb-2023</sub>

    <h4>Release Version 1.0</h4>
    We have implemented GPT-zero to detect if a text is generated from A.I. <br>
    <sub>27-Jan-2023</sub>
    """

def discordHTML():
    return """
    <a href="https://discord.com/invite/F3kFan28vH">
    <button style="border-radius: 5px;font-size: 16px;color: #FFFFFF;background-color: #7289DA;text-align:center;padding:0.5em;item-align:center;">
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="50" viewBox="0 0 16 16" style="item-align:center;">
        <path fill="#FFFFFF", d="M13.545 2.907a13.227 13.227 0 0 0-3.257-1.011.05.05 0 0 0-.052.025c-.141.25-.297.577-.406.833a12.19 12.19 0 0 0-3.658 0 8.258 8.258 0 0 0-.412-.833.051.051 0 0 0-.052-.025c-1.125.194-2.22.534-3.257 1.011a.041.041 0 0 0-.021.018C.356 6.024-.213 9.047.066 12.032c.001.014.01.028.021.037a13.276 13.276 0 0 0 3.995 2.02.05.05 0 0 0 .056-.019c.308-.42.582-.863.818-1.329a.05.05 0 0 0-.01-.059.051.051 0 0 0-.018-.011 8.875 8.875 0 0 1-1.248-.595.05.05 0 0 1-.02-.066.051.051 0 0 1 .015-.019c.084-.063.168-.129.248-.195a.05.05 0 0 1 .051-.007c2.619 1.196 5.454 1.196 8.041 0a.052.052 0 0 1 .053.007c.08.066.164.132.248.195a.051.051 0 0 1-.004.085 8.254 8.254 0 0 1-1.249.594.05.05 0 0 0-.03.03.052.052 0 0 0 .003.041c.24.465.515.909.817 1.329a.05.05 0 0 0 .056.019 13.235 13.235 0 0 0 4.001-2.02.049.049 0 0 0 .021-.037c.334-3.451-.559-6.449-2.366-9.106a.034.034 0 0 0-.02-.019Zm-8.198 7.307c-.789 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.45.73 1.438 1.613 0 .888-.637 1.612-1.438 1.612Zm5.316 0c-.788 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.451.73 1.438 1.613 0 .888-.631 1.612-1.438 1.612Z"/>
    </svg>
    Join Now
    </button>
    </a>
    """

def bannerHTML():
    banner_text = "SG-GPTZero Version 1.1 – We have revamped the entire implementation. Experience new features and improved detection accuracy. We love your feedback !"
    return f'<h4 style="text-align:center;border-radius: 25px;color:#F8F8FF;font-size:1.5em;padding:10px;background-image: linear-gradient(90deg, #20B2AA, #FFEBCD);border-style : solid; border-color : white; border-width: 1px">{banner_text}</h4>'

def emailHTML():
    return """
    <form action="/postdb" method="post" id="formPost">
        <input data-testid="textbox" name="email" type="email" class="scroll-hide block gr-box gr-input w-full gr-text-input" placeholder="Enter your email" autocomplete="email">
        <button class="gr-button gr-button-lg gr-button-secondary"  type="submit" form="formPost" id="email" style="margin-top:10px;">Send</button>
    </form>
    """
