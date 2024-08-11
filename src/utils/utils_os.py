import os


def factory_fct_linked_path(ROOT_DIR, path_to_folder):
    """
    Semantics:

    Args:
        ROOT_DIR: path to the root of the project.
        path_to_folder: a path written in the format you want because we use the function os.path.join to link it.

    Returns:
        The linker
    Examples:
              linked_path = factory_fct_linked_path(ROOT_DIR, "path/a"):
              path_save_history = linked_path(['plots', f"best_score_{nb}.pth"])
              #and ROOT_DIR should be imported from a script at the root where it is written:

              import os
              ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    """
    # example:

    PATH_TO_ROOT = os.path.join(ROOT_DIR, path_to_folder)

    def linked_path(path):
        # a list of folders like: ['C','users','name'...]
        # when adding a '' at the end like
        #       path_to_directory = linker_path_to_result_file([path, ''])
        # one adds a \ at the end of the path. This is necessary in order to continue writing the path.
        return os.path.join(PATH_TO_ROOT, *path)

    return linked_path


def rmv_file(file_path):
    """
    Semantics:
        Wrapper around os to remove a file. It will call remove only if they file exists, nothing otherwise.

    Args:
        file_path: The full path to the file.

    Returns:
        Void.
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        print(f"File {file_path} does not exist or is not a file. File not removed.")
