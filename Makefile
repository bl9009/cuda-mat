build:
	@cd build && cmake --build . --config Release

regen:
	@cd build && cmake -A x64 ..
	@cd build && cmake --build . --target install --config Release

install:
	@cd dist && @copy ..\python\src\pycudamat.py .
	@cd dist && python setup.py install

test:
	@cd python && python -m unittest discover

build-install:
	$(MAKE) build && $(MAKE) install

full:
	$(MAKE) regen && $(MAKE) install

clean:
	@echo cleaning working directory...

	@rd /s /q build && @md build

	@cd dist\build && @rd /s /q lib && @md lib

	@echo done!
