# Makefile
# convert .jl to .ipynb to .html

# key commands:
root = ~/.julia/conda/3/bin/
jt = $(root)/jupytext
jup = $(root)/jupyter
# --allow-errors
juprun = $(jup) nbconvert --execute \
	--config=config/jupyter_nbconvert_config.py
juphtm = $(juprun) --to HTML

# https://nbconvert.readthedocs.io/
# https://nbconvert.readthedocs.io/en/latest/usage.html
# https://nbconvert.readthedocs.io/en/latest/execute_api.html?highlight=execute#module-nbconvert.preprocessors

jl := $(wildcard */*.jl)
ip := $(jl:%.jl=%.ipynb)
ht := $(jl:%.jl=%.html)


all: $(ip) $(ht)


$(ip): %.ipynb: %.jl
	$(jt) --to notebook $? # -o $@


$(ht): %.html: %.ipynb
	$(juphtm) $? # --output $@
