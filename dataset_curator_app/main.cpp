#include <QApplication>
#include <QDir>
#include <QMessageBox>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    if (argc < 2) {
        QMessageBox::critical(nullptr, "Erro",
            "Uso: curator_app <diretorio_raiz>");
        return 1;
    }

    QDir rootDir(argv[1]);
    if (!rootDir.exists()) {
        QMessageBox::critical(nullptr, "Erro",
            "Diretório raiz inválido.");
        return 1;
    }

    MainWindow w(rootDir.absolutePath());
    w.show();

    return app.exec();
}

