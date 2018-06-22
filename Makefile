SEED=0

all: data_directories figure1 figure2 figure3 figure4 figure5 figure6

data_directories: data data/mult_many_trigger_producthp data/periodic_trigger_besthp data/productm_testfiltered_random data/random_bound_follow data/random_bound_once data/random_bound_periodic data/random_bound_random data/random_testfiltered_random

data:	
	mkdir data

data/%:
	mkdir $@

figure1: img/figure1.pdf
img/figure1.pdf: src/combined_store.py
	src/combined_store.py $(SEED)
	mv combined_store_50_$(SEED).pdf img/figure1.pdf

figure2: img/figure2.pdf
img/figure2.pdf: data/random_testfiltered_random/$(SEED)_random_testfiltered_random_abs_error.npy src/draw_memory_task.py
	src/draw_memory_task.py data/random_testfiltered_random/$(SEED)_random_testfiltered_random_inputs.npy data/random_testfiltered_random/$(SEED)_random_testfiltered_random_outputs.npy data/random_testfiltered_random/$(SEED)_random_testfiltered_random_desired_outputs.npy -o img/figure2.pdf --log -i 1
data/random_testfiltered_random/%_random_testfiltered_random_abs_error.npy: src/random_testfiltered_random.py
	src/random_testfiltered_random.py $* -o $@ -r data/random_testfiltered_random/$*_random_testfiltered_random_outputs.npy -d data/random_testfiltered_random/$*_random_testfiltered_random_desired_outputs.npy -i data/random_testfiltered_random/$*_random_testfiltered_random_inputs.npy

figure3: img/figure3.pdf
img/figure3.pdf: data/random_bound_random/$(SEED)_random_bound_random_abs_error.npy data/random_bound_once/$(SEED)_random_bound_once_abs_error.npy data/random_bound_periodic/$(SEED)_random_bound_periodic_abs_error.npy data/random_bound_follow/$(SEED)_random_bound_follow_abs_error.npy src/boxplotsimplified_error_drift_all.py
	src/boxplotsimplified_error_drift_all.py data/random_bound_once/$(SEED)_random_bound_once_abs_error.npy data/random_bound_periodic/$(SEED)_random_bound_periodic_abs_error.npy data/random_bound_follow/$(SEED)_random_bound_follow_abs_error.npy -o img/figure3.pdf
data/random_bound_random/%_random_bound_random_abs_error.npy: bound_random_random.py
	src/bound_random_random.py $* -o $@
data/random_bound_once/%_random_bound_once_abs_error.npy: bound_random_once.py
	src/bound_random_once.py $* -o $@
data/random_bound_periodic/%_random_bound_periodic_abs_error.npy: bound_random_periodic.py
	src/bound_random_periodic.py $* -o $@
data/random_bound_follow/%_random_bound_follow_abs_error.npy: bound_random_follow.py
	src/bound_random_follow.py $* -o $@

figure4: data/periodic_trigger_besthp/$(SEED)_components.npy
data/periodic_trigger_besthp/%_components.npy: data/periodic_trigger_besthp/$(SEED)_internals.npy
	src/pc.py -x data/periodic_trigger_besthp/$*_internals.npy -c data/periodic_trigger_besthp/$*_components.npy -e data/periodic_trigger_besthp/$*_abs_eigs.npy
data/periodic_trigger_besthp/%_internals.npy: src/periodic_trigger_besthp.py
	src/periodic_trigger_besthp.py $(SEED) -i data/periodic_trigger_besthp/$(SEED)_inputs.npy -d data/periodic_trigger_besthp/$(SEED)_desired_outputs.npy -x data/periodic_trigger_besthp/$(SEED)_internals.npy -o data/periodic_trigger_besthp/$(SEED)_outputs.npy -e data/periodic_trigger_besthp/$(SEED)_abs_error.npy

figure5: img/figure5.pdf
img/figure5.pdf: data/productm_testfiltered_random/$(SEED)_productm_testfiltered_random_abs_error.npy src/draw_product_task_test.py
	src/draw_product_task_test.py data/productm_testfiltered_random/$(SEED)_productm_testfiltered_random_inputs.npy data/productm_testfiltered_random/$(SEED)_productm_testfiltered_random_outputs.npy data/productm_testfiltered_random/$(SEED)_productm_testfiltered_random_desired_outputs.npy -o img/figure5.pdf -i 1
data/productm_testfiltered_random/%_productm_testfiltered_random_abs_error.npy: src/productm_testfiltered_random.py
	src/productm_testfiltered_random.py $* -o $@ -r data/productm_testfiltered_random/$*_productm_testfiltered_random_outputs.npy -d data/productm_testfiltered_random/$*_productm_testfiltered_random_desired_outputs.npy -i data/productm_testfiltered_random/$*_productm_testfiltered_random_inputs.npy

figure6: data/mult_many_trigger_producthp/$(SEED)_components.npy
data/mult_many_trigger_producthp/%_components.npy: data/mult_many_trigger_producthp/$(SEED)_internals.npy
	src/pc.py -x data/mult_many_trigger_producthp/$*_internals.npy -c data/mult_many_trigger_producthp/$*_components.npy -e data/mult_many_trigger_producthp/$*_abs_eigs.npy
data/mult_many_trigger_producthp/%_internals.npy: src/mult_many_trigger_producthp.py
	src/mult_many_trigger_producthp.py $(SEED) -i data/mult_many_trigger_producthp/$(SEED)_inputs.npy -d data/mult_many_trigger_producthp/$(SEED)_desired_outputs.npy -x data/mult_many_trigger_producthp/$(SEED)_internals.npy -o data/mult_many_trigger_producthp/$(SEED)_outputs.npy -e data/mult_many_trigger_producthp/$(SEED)_abs_error.npy

.PHONY: figure1 figure2 figure3 figure4 figure5 figure6
