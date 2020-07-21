import os
import shutil
import subprocess

import matplotlib.pyplot as plt

PAPERDIR = "/home/mwolf/src/ptycho-NCA-paper/ptychography"
# COLSIZE = 3.25 # inches
# SUPP_SIZE = 4.7737

TEX_TEMPLATE = r"""
\RequirePackage{luatex85}
\documentclass{standalone}

\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgf}
\usepackage{siunitx}
\usepackage[version=4]{mhchem}

\newcommand{\nca}[1]{\ce{Li_{#1}Ni_{0.8}Co_{0.15}Al_{0.05}O_2}}
\newcommand{\mahg}[1]{\si{\milli\ampere\hour\per\gram}}
\newcommand{\xrdlabel}{$\lvert \vec{q} \rvert$ / \si{\per\angstrom}}

\begin{document}
\textsf{%
"""

TEX_END = r"""
}%
\end{document}
"""


def savefig(fname, *args, **kwargs):
    return savepdf(fname, *args, **kwargs)


def savesuppfig(fname, *args, **kwargs):
    pdf = savepdf(fname, *args, **kwargs)
    base, ext = os.path.splitext(pdf)
    # Now convert to a jpeg
    # response = subprocess.run(['pdftoppm', '-jpeg', '-singlefile', '-r', '300', pdf, base])
    # os.remove(pdf)
    return 


def savepdf(fname, fig=None, debug=False, *args, **kwargs):
    if fig is None:
        fig = plt.gcf()
    # Parse the filename and path
    path, ext = os.path.splitext(fname)
    dirname, base = os.path.split(path)
    # Save the file as PDF
    fig.savefig('{}.pgf'.format(base), *args, **kwargs)
    # Create the tex document
    # os.chdir(dirname)
    tex = ''.join(TEX_TEMPLATE + '\input{' + base + '.pgf}' + TEX_END)
    texname = '{}.tex'.format(base)
    with open(texname, mode='w') as fp:
        fp.write(tex)
    # Compile the tex document
    cmds = ['lualatex', texname]
    response = subprocess.run(cmds, encoding='utf-8',
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if debug:
        print(response)
    # Move the new file to the tex directory
    pdfname = '{}.pdf'.format(base)
    new_pdfname  = os.path.join(dirname, pdfname)
    shutil.move(pdfname, new_pdfname)
    # Clean up temporary files if tex compiled successfully
    if response.returncode == 0 and not debug:
        os.remove('{}.pgf'.format(base))
        os.remove('{}.tex'.format(base))
    return new_pdfname

def saveeps(fname, *args, **kwargs):
    # Parse the filename and path
    path, ext = os.path.splitext(fname)
    dirname, base = os.path.split(path)
    # First compile to PDF
    pdfname = '{}.pdf'.format(base)
    savepdf(pdfname, *args, **kwargs)
    # Now convert the PDF to EPS
    epsname = '{}.eps'.format(base)
    cmds = ['pdftops', '-eps', pdfname, epsname]
    response = subprocess.run(cmds, encoding='utf-8',
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Move the new file to the tex directory
    shutil.move(epsname, os.path.join(dirname, epsname))
