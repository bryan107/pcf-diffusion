import os

from matplotlib import pyplot as plt


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
    return


def makedir(directory_where_to_save):
    if not os.path.exists(directory_where_to_save):
        if directory_where_to_save != "":
            os.makedirs(directory_where_to_save)
    return


def remove_files_from_dir(
    folder_path: str, file_start: str = "", file_extension: str = ""
) -> None:
    """
    Remove all files from a folder with a certain beginning or ending in their name.

    Args:
        folder_path (str): The path to the folder.
        file_start (str): The beginning of the name of the files to be deleted.
        file_extension (str): The ending of the name of the files to be deleted.

    Returns:
        None
    """
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.startswith(file_start) and file.endswith(file_extension):
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
    else:
        print(f"Folder {folder_path} does not exist.")
    return


def remove_file(file_path) -> None:
    """
    Semantics:
        Wrapper around os to remove a file. It will call remove only if they file exists, nothing otherwise.

    Args:
        file_path: The full path to the file.
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def savefig(fig: plt.Figure, path_file: str) -> None:
    """
    Saves a matplotlib figure to the specified file path.

    Args:
        fig (plt.Figure): The matplotlib figure to save.
        path_file (str): The full path to the file where the figure should be saved.
                         The path should include the file extension, for example .png but not mandatory (png default).

    More information @https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

    Returns:
        None
    """
    directory_where_to_save = os.path.dirname(path_file)
    makedir(directory_where_to_save)
    fig.savefig(path_file)
    return
