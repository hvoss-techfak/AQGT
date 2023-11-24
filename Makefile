
build:
	@echo $(BUILD_MESSAGE)
	docker build . -t aqgt
run:
	docker run --ipc=host -v ./dataset:/home/appuser/AQ-GT/dataset -v ./config:/home/appuser/AQ-GT/config -v ./test_full:/home/appuser/AQ-GT/test_full --gpus all -it --rm aqgt
