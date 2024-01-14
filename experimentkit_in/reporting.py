""" Reporting utilities """
from pathlib import Path

class ReportMD():
    
    def __init__(self, md_path:str, title: str = None, create: bool = True):
        md_path = Path(md_path)
        if not md_path.exists():
            if not create:
                raise FileNotFoundError(f"md_path does not exist")
            md_path.write_text("")  # create a new file
        
        self.md_path = md_path
        self.write_title(title)
        return
    
    def clean_all_the_text(self, imsure: bool = False):
        if imsure:
            self.md_path.write_text("")
        return

    def add_txt(self, txt: str, new_line: bool = True) -> None:
        "Append a text"
        txt_ = txt
        if new_line:
            txt_ = "  \n" + txt
        with open(self.md_path, 'a+') as f:
            f.write(txt_)
        return
    
    def add_code(self, txt: str, language: str, new_line: bool = True) -> None:
        txt_ = "\n".join([
            f"```{language}",
            txt,
            "```"
        ])
        self.add_txt(txt_)
        return
    
    def write_title(self, title: str) -> None:
        if title is None:
            raise ValueError("A title must be provided")

        with open(self.md_path, 'r') as f:
            for line in f.readlines():
                line_ = line.replace("\n", "").strip()
                if len(line_) > 0:
                    if line_.startswith("# "):
                        return  # has a title
                    break  # hasn't a title
        
        with open(self.md_path, 'w+') as f:
            title_ = f"# {title}\n".title()
            f.write(title_ + f.read())
        return
    
    def add_img(self, img_path: str, more: str = '') -> None:
        "Append an image"
        ipath = Path(img_path)
        if not ipath.exists():
            raise FileNotFoundError
        txt = f"<img src='{img_path}' {more}>"
        self.add_txt(txt)


            

