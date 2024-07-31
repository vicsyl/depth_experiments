import os
import shutil
from collections import defaultdict
from pathlib import Path


class HtmlGallery:

    def __init__(self,
                 types_for_key=None,
                 filter_out=["erosion", "intersection"],
                 use_p=True,
                 columns_by_width=False):

        # FIXME not used
        self.types_for_key = types_for_key
        self.filter_out = filter_out
        self.use_p = use_p
        self.columns_by_width = columns_by_width

    def filter_out_fn(self, file_name):
        for f in self.filter_out:
            if file_name.__contains__(f):
                return True
        return False

    def get_file_name_key(self, file_name):
        index = file_name.rfind(".")
        if index == -1 or file_name[index + 1:].lower() not in ["jpg", "png"]:
            return None
        if not file_name.__contains__("."):
            return None
        index = file_name.index(".")
        if index == -1:
            index = len(file_name)
        key = file_name[:index]
        return key

    def write_header(self, fd, gallery_name):
        fd.write(f"""<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>{gallery_name}></title>
                </head>
                <body>\n""")

    def write_gallery_file(self,
                           in_folder,
                           target_folder,
                           gallery_name=None,
                           width_pc=19,
                           html_file_name="index.html",
                           max_keys=100000,
                           purge_target=False):

        """
         -- target_folder == in_folder
            |---img_key1.img1.jpg  # img_key1.* and img_key2.* will be on the same row
            |---img_key1.img2.jpg
            |---img_key2.img1.jpg
            |---img_key2.img2.jpg
            |---index.html

        OR

         -- in_folder copied to

         -- target_folder
            |---imgs/
                |---img1.jpg
                |---img2.jpg
            |---index.html

        :param target_folder:
        :param gallery_name:
        :param width_pc:
        :param html_file_name:
        :param max_keys:
        :param purge_target:
        :return:
        """

        if not gallery_name:
            gallery_name = f"Gallery from {in_folder}"

        in_place = target_folder == in_folder

        if purge_target and not in_place:
            shutil.rmtree(target_folder, ignore_errors=True)

        if in_place:
            print(f"Will create the gallery index.html in place to {target_folder}")
            whole_folder = target_folder
            rel_img_folder = "./"
        else:
            whole_folder = f"{target_folder}/imgs"
            Path(whole_folder).mkdir(parents=True, exist_ok=True)
            rel_img_folder = "./imgs/"

        files_map = defaultdict(list)
        for file_name in sorted(os.listdir(in_folder)):
            if self.filter_out_fn(file_name):
                continue
            key = self.get_file_name_key(file_name)
            if not key:
                continue

            if max_keys <= len(files_map) and not files_map.__contains__(key):
                continue

            files_map[key].append(file_name)
            if not in_place:
                shutil.copy(f"{in_folder}/{file_name}", f"{whole_folder}/{file_name}")

        out_file = f"{target_folder}/{html_file_name}"
        with open(out_file, "w") as fd:
            self.write_header(fd, gallery_name)

            counter = 0
            for k, l in files_map.items():
                counter += 1
                if self.types_for_key is None:
                    f_l = l
                else:
                    f_l = []
                    for t in self.types_for_key:
                        f_l.extend([f for f in l if f.__contains__(t)])

                if self.use_p or self.columns_by_width and counter == 100 // width_pc:
                    fd.write(f"<p>{k}</p><p/><p/>")
                for f_n in f_l:
                    fd.write(f"""<a href="{rel_img_folder}{f_n}"><img src="{rel_img_folder}{f_n}" width="{width_pc}%"/></a>\n""")

            fd.write("""</body>
                    </html>\n""")

        print(f"saved to {out_file}")


if __name__ == "__main__":
    #write_gallery_file("data/out/apt1/kitchen")
    # write_gallery_file(folder="data/remote/out/apt1/living",
    #                    target_folder="data/to_show/galls/apt1/living")

    gal = HtmlGallery()
    gal.write_gallery_file(in_folder="data/remote/out/apt1/kitchen",
                           target_folder="data/test/galls/apt1/kitchen",
                           max_keys=10)
