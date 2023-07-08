from argparse import ArgumentParser
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from rich.console import Console
from rich.progress import track

c = Console()


def main(args):
    if not Path(args.src).is_dir():
        c.print(f"[b red]Directory `{args.src}` not found![/]")

    if not (path := Path("debug")).is_dir() and args.debug:
        path.mkdir()

    c.print("[green]Getting needed face...[/]")

    input_face = face_recognition.load_image_file(args.face)
    input_face_encoding = face_recognition.face_encodings(input_face)

    c.print("[green b]Scanning `{}` dir...[/]".format(args.src))
    for faces_image in track(
        list(Path(args.src).iterdir()),
        console=c,
        description="[b green]Detecting faces...[/]",
    ):
        c.print(f"[yellow b]Working with[/] [cyan]{faces_image}[/]")
        faces = face_recognition.load_image_file(str(faces_image))
        pil_image = Image.fromarray(faces)
        draw = ImageDraw.Draw(pil_image)
        faces_locations = face_recognition.face_locations(faces)
        faces_encoding = face_recognition.face_encodings(faces)
        needed_face = False
        for (top, right, bottom, left), face_encoding in zip(
            faces_locations, faces_encoding
        ):
            matches = face_recognition.compare_faces(input_face_encoding, face_encoding)
            face_distances = face_recognition.face_distance(
                input_face_encoding, face_encoding
            )

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                needed_face = True

            if needed_face:
                c.print("[green b]Face found! Writing to output...[/]")
                cropped_image = pil_image.copy().crop(
                    (
                        max(0, left - 64),
                        max(0, top - 64),
                        min(pil_image.width, right + 64),
                        min(pil_image.height, bottom + 64),
                    )
                )
                cropped_image.thumbnail((512, 512))
                cropped_image = cropped_image.resize((512, 512))
                cropped_image.save(Path(args.dst) / faces_image.name)
                break
            else:
                c.print("[yellow]Detecting face...[/]")

            if args.debug:
                draw.rectangle(
                    ((left, top), (right, bottom)),
                    outline=(0, 255, 0) if needed_face else (0, 0, 255),
                    width=1,
                )
                text = "DETECTED" if needed_face else "UNKNWN"

                # _, text_height = draw.textsize(text)

                _ = draw.textbbox((0, 0), text)
                text_height = _[2] - _[0]
                draw.rectangle(
                    ((left, bottom - text_height), (right, bottom)),
                    fill=(0, 255, 0) if needed_face else (0, 0, 255),
                    outline=(0, 255, 0) if needed_face else (0, 0, 255),
                )
                draw.text(
                    (left + 6, bottom - text_height - 5),
                    text,
                    fill=(255, 255, 255, 255),
                )
        if args.debug:
            pil_image.save(Path("debug") / faces_image.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--src", required=True, help="Directory with images")
    parser.add_argument("-f", "--face", required=True, help="Face image")
    parser.add_argument(
        "-o", "--dst", required=True, default="output", help="Folder for faces"
    )
    parser.add_argument(
        "-d",
        "--debug",
        required=False,
        help="Debug mode. Write recognizing to `debug` folder",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
