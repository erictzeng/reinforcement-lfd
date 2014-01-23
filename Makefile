.PHONY: drypush push drypull pull

DATE := $(shell date "+%Y-%m-%d-%H:%M:%S")
DATADIR := /media/hdd/data/reinforcement-lfd
PRIMUS := primus.banatao.berkeley.edu

ifdef DELETE
	delflag = --delete
else
	delflag =
endif

drypush:
	@rsync -rlvz --dry-run $(delflag) data/ $(PRIMUS):$(DATADIR)/data/ | grep -E '^deleting|[^/]$$'

push:
	@ssh $(PRIMUS) "rsync -rlz --link-dest=$(DATADIR)/backup/most-recent $(DATADIR)/data/ $(DATADIR)/backup/back-$(DATE); rm -f $(DATADIR)/backup/most-recent && ln -s back-$(DATE) $(DATADIR)/backup/most-recent"
	@rsync -rlzP $(delflag) data/ $(PRIMUS):$(DATADIR)/data/

drypull:
	@rsync -avz --dry-run $(delflag) $(PRIMUS):$(DATADIR)/data/ data/ | grep -E '^deleting|[^/]$$'

pull:
	@rsync -azP $(delflag) $(PRIMUS):$(DATADIR)/data/ data/
