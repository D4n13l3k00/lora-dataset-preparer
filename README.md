# Dataset preparing utility for LoRA

## Usage

```shell
poetry install
poetry run python main.py -i INPUT_IMAGES_DIR -f FACE_IMAGE_FOR_DETECTING -o OUTPUT_FOLDER 
```

This will be detect `-f` face in all images in `-i` directory, then cut face and resize into `-o` folder

## License

This project under [GNU General Public License v3.0](/LICENSE)
