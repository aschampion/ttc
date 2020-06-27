SHELL := /bin/bash

check-slabs:
ifndef slabs
	$(error slabs is undefined)
endif

dir-setup:
	ln -s ../../Makefile
	ln -s ../log
	ln -s /data1/championa/Tissue\ labeling data
	python ~/code/ttc-project/docker/ttc/mknet.py

ttc-train:
	CUDA_VISIBLE_DEVICES=0 python ~/code/ttc-project/docker/ttc/gunpowder_ttc.py $$(basename "$$PWD")

ttc-predict: check-slabs
	mkdir -p predictions.n5/volumes/labels/ttc
	cp -n ../../template_attributes.json predictions.n5/volumes/labels/ttc/attributes.json || true
	parallel --no-notice --ungroup CUDA_VISIBLE_DEVICES={1} python ~/code/ttc-project/docker/ttc/gunpowder_ttc_predict.py {1} $(slabs) :::: <(seq 0 $$(($(slabs) - 1)))

nuclei-train:
	CUDA_VISIBLE_DEVICES=0 python ~/code/ttc-project/docker/ttc/gunpowder_nuclei.py $$(basename "$$PWD")

nuclei-predict: check-slabs
	mkdir -p predictions.n5/volumes/labels/nuclei
	cp -n ../../template_attributes.json predictions.n5/volumes/labels/nuclei/attributes.json || true
	parallel --no-notice --ungroup CUDA_VISIBLE_DEVICES={1} python ~/code/ttc-project/docker/ttc/gunpowder_nuclei_predict.py {1} $(slabs) :::: <(seq 0 $$(($(slabs) - 1)))

