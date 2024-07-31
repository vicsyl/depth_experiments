from gallery import HtmlGallery


def one_off_hist_gallery():
    dir = "./data/histograms"
    HtmlGallery().write_gallery_file(in_folder=dir,
                                     target_folder=dir,
                                     width_pc=19)


if __name__ == "__main__":
    one_off_hist_gallery()
