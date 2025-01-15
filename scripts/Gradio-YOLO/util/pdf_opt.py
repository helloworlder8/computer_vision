from fpdf import FPDF


# PDF生成类
class PDF(FPDF):
    # 参考：https://pyfpdf.readthedocs.io/en/latest/Tutorial/index.html
    def header(self):
        # 设置中文字体
        self.add_font("SimSun", "", "./fonts/SimSun.ttf", uni=True)
        self.set_font("SimSun", "", 16)
        # Calculate width of title and position
        w = self.get_string_width(title) + 6
        self.set_x((210 - w) / 2)
        # Colors of frame, background and text
        self.set_draw_color(255, 255, 255)
        self.set_fill_color(255, 255, 255)
        self.set_text_color(0, 0, 0)
        # Thickness of frame (1 mm)
        # self.set_line_width(1)
        # Title
        self.cell(w, 9, title, 1, 1, "C", 1)
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # 设置中文字体
        self.add_font("SimSun", "", "./fonts/SimSun.ttf", uni=True)
        self.set_font("SimSun", "", 12)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "C")

    def chapter_title(self, num, label):
        # 设置中文字体
        self.add_font("SimSun", "", "./fonts/SimSun.ttf", uni=True)
        self.set_font("SimSun", "", 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        # self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        self.cell(0, 6, "检测结果：", 0, 1, "L", 1)
        # Line break
        self.ln(4)

    def chapter_body(self, name):

        # 设置中文字体
        self.add_font("SimSun", "", "./fonts/SimSun.ttf", uni=True)
        self.set_font("SimSun", "", 12)
        # Output justified text
        self.multi_cell(0, 5, name)
        # Line break
        self.ln()
        self.cell(0, 5, "--------------------------------------")

    def print_chapter(self, num, title, name):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(name)

    def print_img(self, name, x, y, width):
        self.add_page()
        self.image(name, x, y, width)


# pdf生成函数
def pdf_generate(input_file, origin_det_imgs, output_file, title_):
    global title

    title = title_
    pdf = PDF()
    pdf.set_title(title)
    pdf.set_author("Zeng Yifu")
    pdf.print_img(origin_det_imgs[0], 10, 30, 150)
    pdf.print_img(origin_det_imgs[1], 10, 30, 150)
    pdf.print_chapter(1, "A RUNAWAY REEF", input_file)
    pdf.output(output_file)
