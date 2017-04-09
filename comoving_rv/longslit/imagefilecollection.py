# Standard library
from fnmatch import fnmatch

# Third-party
from ccdproc import ImageFileCollection
import six

__all__ = ['GlobImageFileCollection']

class GlobImageFileCollection(ImageFileCollection):

    def __init__(self, location=None, keywords=None, info_file=None,
                 filenames=None, glob_include=None, glob_exclude=None,
                 skip_filenames=None):

        if skip_filenames is None:
            self.skip_filenames = list()

        else:
            self.skip_filenames = list(skip_filenames)

        self.glob_exclude = glob_exclude
        self.glob_include = glob_include

        super(GlobImageFileCollection, self).__init__(location=location,
                                                      keywords=keywords,
                                                      info_file=info_file,
                                                      filenames=filenames)

    def _get_files(self):
        """ Helper method which checks whether ``files`` should be set
        to a subset of file names or to all file names in a directory.
        Returns
        -------
        files : list or str
            List of file names which will be added to the collection.
        """
        files = []
        if self._filenames:
            if isinstance(self._filenames, six.string_types):
                files.append(self._filenames)
            else:
                files = self._filenames
        else:
            _files = self._fits_files_in_directory()

            files = []
            for fn in _files:
                if fn in self.skip_filenames:
                    continue

                # logic is backwards because we continue if fnmatch() doesn't evaluate
                if self.glob_include is not None and not fnmatch(fn, self.glob_include):
                    continue

                if self.glob_exclude is not None and fnmatch(fn, self.glob_exclude):
                    continue

                files.append(fn)

        return files
