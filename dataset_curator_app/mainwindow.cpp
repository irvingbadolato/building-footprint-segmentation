#include "mainwindow.h"

#include <QDebug>
#include <QDir>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QFile>
#include <QRegularExpression>
#include <QPainter>
#include <QShortcut>
#include <QTextStream>
#include <QMouseEvent>
#include <QDockWidget>
#include <QStatusBar>
#include <QToolBar>
#include <QMenuBar>
#include <QScreen>
#include <QWindow>

#include <variant>
#include <algorithm>
#include <gdal_priv.h>
#define DATASET_SIZE 384
#define STEP_SIZE 45

using Token = std::variant<int, QString>;

QVector<Token> tokenize(const QString& s)
{
    static QRegularExpression re("[._]");
    QStringList parts = s.split(re, Qt::SkipEmptyParts);

    QVector<Token> tokens;
    tokens.reserve(parts.size());

    for (const QString& p : parts) {
        bool ok = false;
        int val = p.toInt(&ok);
        if (ok) {
            tokens.emplace_back(val);      // parte numérica
        } else {
            tokens.emplace_back(p);        // parte textual
        }
    }
    return tokens;
}

bool naturalLess(const QString& a, const QString& b)
{
    auto ta = tokenize(a);
    auto tb = tokenize(b);

    int n = std::min(ta.size(), tb.size());

    for (int i = 0; i < n; ++i) {
        const auto& A = ta[i];
        const auto& B = tb[i];

        if (A.index() != B.index()) {
            return A.index() < B.index();
        }

        if (std::holds_alternative<int>(A)) {
            int ia = std::get<int>(A);
            int ib = std::get<int>(B);
            if (ia != ib) return ia < ib;
        } else {
            const QString& sa = std::get<QString>(A);
            const QString& sb = std::get<QString>(B);
            if (sa != sb) return sa < sb;
        }
    }

    // desempate por tamanho
    return ta.size() < tb.size();
}

MainWindow::MainWindow(const QString& root, QWidget *parent)
    : QMainWindow(parent), rootPath(root)
{
    GDALAllRegister();

    // ---------- Construct UI elements ----------
    datasetCombo = new QComboBox;
    filterCombo  = new QComboBox;
    filterCombo->addItems({
        "Todas as imagens",
        "Somente selecionadas",
        "Somente não selecionadas"
    });

    selectCheck = new QCheckBox("");

    indexEdit = new QLineEdit;
    indexEdit->setFixedWidth(90);
    indexEdit->setAlignment(Qt::AlignCenter);

    prevBtn = new QPushButton("◀");
    nextBtn = new QPushButton("▶");

    alphaThreshold = brightness = contrast = (STEP_SIZE-1)/2;

    alphaSlider = new QSlider(Qt::Vertical);
    alphaSlider->setRange(0,STEP_SIZE-1);
    alphaSlider->setValue(alphaThreshold);
    alphaSlider->setTickInterval(alphaThreshold);
    alphaSlider->setTickPosition(QSlider::TicksLeft);
    alphaSlider->setToolTip("Mistura");

    brightnessSlider = new QSlider(Qt::Vertical);
    brightnessSlider->setRange(0,STEP_SIZE-1);
    brightnessSlider->setValue(brightness);
    brightnessSlider->setTickInterval(brightness);
    brightnessSlider->setTickPosition(QSlider::TicksLeft);
    brightnessSlider->setToolTip("Brilho");

    contrastSlider = new QSlider(Qt::Vertical);
    contrastSlider->setRange(0,STEP_SIZE-1);
    contrastSlider->setValue(contrast);
    contrastSlider->setTickInterval(contrast);
    contrastSlider->setTickPosition(QSlider::TicksLeft);
    contrastSlider->setToolTip("Contraste");

    imageLayout = nullptr;
    imageView = new QWidget;
    imageView->setFocusPolicy(Qt::StrongFocus);

    mainViewer = new QScrollArea;
    mainViewer->setWidgetResizable(true);
    mainViewer->setFrameShape(QFrame::NoFrame);
    mainViewer->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    mainViewer->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    mainViewer->setWidget(imageView);

    imgRGB   = new QLabel(imageView);
    imgRGBA  = new QLabel(imageView);
    imgLabel = new QLabel(imageView);
    imgRGB->setFixedSize(DATASET_SIZE,DATASET_SIZE);
    imgRGBA->setFixedSize(DATASET_SIZE,DATASET_SIZE);
    imgLabel->setFixedSize(DATASET_SIZE,DATASET_SIZE);
    imgRGB->installEventFilter(this);
    imgRGBA->installEventFilter(this);
    imgLabel->installEventFilter(this);

    statsLabel = new QLabel;

    spinR = new QSpinBox;
    spinG = new QSpinBox;
    spinB = new QSpinBox;
    for (auto *s : {spinR, spinG, spinB}) {
        s->setMinimum(1);
        s->setValue(1);
    }
    spinG->setValue(2);
    spinB->setValue(3);

    normSelector = new QCheckBox("Ver bandas normalizadas");

    gridViewer = new GridViewer;
    gridViewer->resize(DATASET_SIZE*3,DATASET_SIZE*3);

    dockArea = new QDockWidget("Imagens adjacentes");
    dockArea->setMinimumSize(DATASET_SIZE*3,DATASET_SIZE);
    dockArea->setWidget(gridViewer);
    dockArea->setAllowedAreas(Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
    dockArea->setHidden(true);

    addDockWidget(Qt::BottomDockWidgetArea, dockArea);

    // ---------- Tool Bar ----------
    QWidget *image_navigator = new QWidget;
    QHBoxLayout *in_layout = new QHBoxLayout(image_navigator);
    in_layout->addWidget(prevBtn);
    in_layout->addWidget(indexEdit);
    in_layout->addWidget(nextBtn);

    QWidget *lspacer = new QWidget, *rspacer = new QWidget, *bspacer = new QWidget;
    lspacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    rspacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    bspacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QToolBar *navbar = addToolBar("Seletor de imagens");
    navbar->setAllowedAreas(Qt::TopToolBarArea | Qt::BottomToolBarArea);
    navbar->addWidget(selectCheck);
    navbar->addWidget(lspacer);
    navbar->addSeparator();
    navbar->addWidget(image_navigator);
    navbar->addSeparator();
    navbar->addWidget(rspacer);
    navbar->addWidget(filterCombo);
    navbar->addWidget(datasetCombo);

    QToolBar *pdi_tool = new QToolBar("Brilho, contraste e mistura");
    pdi_tool->setAllowedAreas(Qt::LeftToolBarArea | Qt::RightToolBarArea);
    pdi_tool->addWidget(brightnessSlider);
    pdi_tool->addWidget(contrastSlider);
    pdi_tool->addWidget(alphaSlider);
    pdi_tool->addWidget(bspacer);
    pdi_tool->setOrientation(Qt::Vertical);
    addToolBar(Qt::LeftToolBarArea, pdi_tool);

    // ---------- Central Widget ----------
    setCentralWidget(mainViewer);
    setVertical(false);

    // ---------- Status Bar ----------
    QWidget *band_selector = new QWidget;
    QHBoxLayout *bs_layout = new QHBoxLayout;
    bs_layout->addWidget(new QLabel("R"));
    bs_layout->addWidget(spinR);
    bs_layout->addWidget(new QLabel("G"));
    bs_layout->addWidget(spinG);
    bs_layout->addWidget(new QLabel("B"));
    bs_layout->addWidget(spinB);
    bs_layout->addWidget(normSelector);
    band_selector->setLayout(bs_layout);

    statusBar()->addWidget(band_selector);
    statusBar()->addPermanentWidget(statsLabel);

    // ---------- Connections ----------
    connect(datasetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onDatasetChanged);

    connect(selectCheck, &QCheckBox::toggled,
            this, &MainWindow::selectionToggled);

    connect(alphaSlider, &QSlider::valueChanged, this, [this](int v) {
        alphaThreshold = v;
        loadCurrentImage();
    });

    connect(brightnessSlider, &QSlider::valueChanged, this, [this](int v) {
        brightness = v;
        loadCurrentImage();
    });

    connect(contrastSlider, &QSlider::valueChanged, this, [this](int v) {
        contrast = v;
        loadCurrentImage();
    });

    connect(filterCombo, &QComboBox::currentIndexChanged, this, [this](int i) {
        filterMode = static_cast<FilterMode>(i);
        rebuildVisibleIndices();
    });

    connect(indexEdit, &QLineEdit::returnPressed, this, [this]() {
        bool ok;
        int idx = indexEdit->text().section('/',0,0).toInt(&ok);
        if (ok) jumpToIndex(idx - 1);
    });

    connect(normSelector, &QCheckBox::toggled,
            this, &MainWindow::loadCurrentImage);

    connect(spinR, &QSpinBox::valueChanged, this, [this](int v) {
        loadCurrentImage();
    });

    connect(spinG, &QSpinBox::valueChanged, this, [this](int v) {
        loadCurrentImage();
    });

    connect(spinB, &QSpinBox::valueChanged, this, [this](int v) {
        loadCurrentImage();
    });

    // ---------- Actions ----------
    createActions();

    scanDatasets();
    loadViewerPresets();
    resizeToFitImageView();
}

void MainWindow::createActions()
{
    atcUpMix = new QAction(this);
    atcUpMix->setShortcut(Qt::Key_Up);
    connect(atcUpMix, &QAction::triggered, this, &MainWindow::upMixture);
    addAction(atcUpMix);

    actDnMix = new QAction(this);
    actDnMix->setShortcut(Qt::Key_Down);
    connect(actDnMix, &QAction::triggered, this, &MainWindow::downMixture);
    addAction(actDnMix);

    actUpBri = new QAction(this);
    actUpBri->setShortcut(QKeySequence(Qt::SHIFT | Qt::Key_Up));
    connect(actUpBri, &QAction::triggered, this, &MainWindow::upBrightness);
    addAction(actUpBri);

    actDnBri = new QAction(this);
    actDnBri->setShortcut(QKeySequence(Qt::SHIFT | Qt::Key_Down));
    connect(actDnBri, &QAction::triggered, this, &MainWindow::downBrightness);
    addAction(actDnBri);

    actUpCon = new QAction(this);
    actUpCon->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Up));
    connect(actUpCon, &QAction::triggered, this, &MainWindow::upContrast);
    addAction(actUpCon);

    actDnCon = new QAction(this);
    actDnCon->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Down));
    connect(actDnCon, &QAction::triggered, this, &MainWindow::downContrast);
    addAction(actDnCon);

    actNext = new QAction(this);
    actNext->setShortcut(Qt::Key_Right);
    connect(actNext, &QAction::triggered, this, &MainWindow::nextImage);
    addAction(actNext);
    connect(nextBtn, &QPushButton::clicked, actNext, &QAction::triggered);

    actPrev = new QAction(this);
    actPrev->setShortcut(Qt::Key_Left);
    connect(actPrev, &QAction::triggered, this, &MainWindow::prevImage);
    addAction(actPrev);
    connect(prevBtn, &QPushButton::clicked, actPrev, &QAction::triggered);

    actToggleSelect = new QAction(this);
    actToggleSelect->setShortcut(Qt::Key_Space);
    connect(actToggleSelect, &QAction::triggered,
            this, [this]() { selectCheck->toggle(); });
    addAction(actToggleSelect);

    actChangeImageViewer = new QAction(this);
    actChangeImageViewer->setShortcut(Qt::Key_Plus);
    connect(actChangeImageViewer, &QAction::triggered, this, [this]() {
        setVertical(!isVertical);
        if (isVertical) {
            dockArea->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
            addDockWidget(Qt::RightDockWidgetArea, dockArea);
        } else {
            dockArea->setAllowedAreas(Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
            addDockWidget(Qt::BottomDockWidgetArea, dockArea);
        }
        resizeToFitImageView();
        loadCurrentImage();
    });
    addAction(actChangeImageViewer);

    actExchangeViewers = new QAction(this);
    actExchangeViewers->setShortcut(Qt::Key_Minus);
    connect(actExchangeViewers, &QAction::triggered, this, [this]() {
        dockArea->setMinimumSize(DATASET_SIZE*3+30, DATASET_SIZE+30);
        if (centralWidget() != gridViewer) {
            takeCentralWidget();
            dockArea->setWindowTitle("Rótulo da imagem ativa");
            dockArea->setWidget(mainViewer);
            dockArea->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable);
            dockArea->setVisible(true);
            setCentralWidget(gridViewer);
            gridViewer->setVisible(true);
            if (isVertical)
                dockArea->setMinimumSize(DATASET_SIZE+30, DATASET_SIZE*3+30);
        } else {
            takeCentralWidget();
            dockArea->setWindowTitle("Imagens adjacentes");
            dockArea->setWidget(gridViewer);
            dockArea->setFeatures(QDockWidget::DockWidgetFloatable |
                                  QDockWidget::DockWidgetMovable |
                                  QDockWidget::DockWidgetClosable);
            setCentralWidget(mainViewer);
            mainViewer->setVisible(true);
        }
        resizeToFitImageView();
        loadCurrentImage();
    });
    addAction(actExchangeViewers);

    actShowGrid = new QAction(this);
    actShowGrid->setShortcut(Qt::Key_0);
    connect(actShowGrid, &QAction::triggered, this, [this]() {
        if (centralWidget() == mainViewer) {
            dockArea->setVisible(dockArea->isHidden());
            resizeToFitImageView();
            loadCurrentImage();
        }
    });
    addAction(actShowGrid);
}

void MainWindow::setVertical(bool orientation)
{
    if (imageLayout && orientation == isVertical) return;
    auto *layout = new QGridLayout;
    if (orientation) {
        layout->addWidget(imgLabel,0,0);
        layout->addWidget(imgRGBA,1,0);
        layout->addWidget(imgRGB,2,0);
    } else {
        layout->addWidget(imgLabel,0,2);
        layout->addWidget(imgRGBA,0,1);
        layout->addWidget(imgRGB,0,0);
    }
    if (imageLayout)
        delete imageLayout;
    imageLayout = layout;
    imageView->setLayout(imageLayout);
    imageView->resize(imageLayout->sizeHint());
    isVertical = orientation;
}


void MainWindow::scanDatasets()
{
    QDir root(rootPath);
    for (const QString& d : root.entryList(QDir::Dirs |
                                           QDir::NoDotAndDotDot)) {
        QDir c(root.filePath(d));
        if (c.exists("images") && c.exists("labels"))
            datasetCombo->addItem(d);
    }
}

void MainWindow::onDatasetChanged(int)
{
    currentDatasetPath =
        QDir(rootPath).filePath(datasetCombo->currentText());

    currentVisiblePos = 0;
    imgRGB->clear();
    imgRGBA->clear();
    imgLabel->clear();
    indexEdit->clear();
    statsLabel->clear();
    loadSelectedFile();
    loadInvalidFile();
    loadPoints();
    loadImageList();
}

void MainWindow::loadSelectedFile()
{
    selectedSet.clear();
    QFile f(QDir(currentDatasetPath).filePath("selected.txt"));
    if (!f.exists()) return;

    f.open(QIODevice::ReadOnly);
    for (const QString& l :
         QString(f.readAll()).split('\n', Qt::SkipEmptyParts))
        selectedSet.insert(l.trimmed());
}

void MainWindow::saveSelectedFile()
{
    QFile f(QDir(currentDatasetPath).filePath("selected.txt"));
    f.open(QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream out(&f);
    for (const QString& s : selectedSet)
        out << s << "\n";
}

void MainWindow::loadInvalidFile()
{
    invalidSet.clear();
    QFile f(QDir(currentDatasetPath).filePath("invalid_images.txt"));
    if (!f.exists()) return;

    f.open(QIODevice::ReadOnly);
    for (const QString& l :
         QString(f.readAll()).split('\n', Qt::SkipEmptyParts))
        invalidSet.insert(l.section(':',0,0).trimmed());
}

void MainWindow::logInvalidImage(const QString& name, const QString& reason)
{
    if (invalidSet.contains(name)) return;

    invalidSet.insert(name);

    QFile f(QDir(currentDatasetPath).filePath("invalid_images.txt"));
    f.open(QIODevice::Append);
    QTextStream out(&f);
    out << name << " : " << reason << "\n";
}

void MainWindow::loadImageList()
{
    QDir imgDir(QDir(currentDatasetPath).filePath("images"));
    QStringList all =
        imgDir.entryList(QStringList() << "*.tif", QDir::Files, QDir::Name);

    std::sort(all.begin(), all.end(), naturalLess);

    imageNames.clear();
    for (const QString& n : all)
        if (!invalidSet.contains(n))
            imageNames << n;

    rebuildVisibleIndices();
}

void MainWindow::rebuildVisibleIndices()
{
    visibleIndices.clear();

    for (int i=0; i<imageNames.size(); ++i) {
        bool sel = selectedSet.contains(imageNames[i]);
        if (filterMode == FilterMode::All ||
           (filterMode == FilterMode::Selected && sel) ||
           (filterMode == FilterMode::Unselected && !sel))
            visibleIndices.push_back(i);
    }

    loadCurrentImage();
}

void MainWindow::jumpToIndex(int idx)
{
    if (idx < 0 || idx >= visibleIndices.size()) return;
    currentVisiblePos = idx;
    loadCurrentImage();
}

void MainWindow::upMixture()
{
    if (alphaSlider->value() < STEP_SIZE-1) {
        alphaSlider->setValue(alphaSlider->value()+1);
    }
}

void MainWindow::downMixture()
{
    if (alphaSlider->value() > 0) {
        alphaSlider->setValue(alphaSlider->value()-1);
    }
}

void MainWindow::upBrightness()
{
    if (brightnessSlider->value() < STEP_SIZE-1) {
        brightnessSlider->setValue(brightnessSlider->value()+1);
    }
}

void MainWindow::downBrightness()
{
    if (brightnessSlider->value() > 0) {
        brightnessSlider->setValue(brightnessSlider->value()-1);
    }
}

void MainWindow::upContrast()
{
    if (contrastSlider->value() < STEP_SIZE-1) {
        contrastSlider->setValue(contrastSlider->value()+1);
    }
}

void MainWindow::downContrast()
{
    if (contrastSlider->value() > 0) {
        contrastSlider->setValue(contrastSlider->value()-1);
    }
}

void MainWindow::nextImage()
{
    if (currentVisiblePos + 1 < visibleIndices.size()) {
        currentVisiblePos++;
        loadCurrentImage();
    }
}

void MainWindow::prevImage()
{
    if (currentVisiblePos > 0) {
        currentVisiblePos--;
        loadCurrentImage();
    }
}

void MainWindow::loadCurrentImage()
{
    selectCheck->setText("no image available");
    if (visibleIndices.isEmpty()) return;

    if (currentVisiblePos >= visibleIndices.size())
        currentVisiblePos = visibleIndices.size() - 1;
    int realIdx = visibleIndices[currentVisiblePos];
    QString name = imageNames[realIdx];

    bool okImg=false, okLbl=false;
    QImage rgb   = loadGDALImage(
        QDir(currentDatasetPath).filePath("images/" + name), okImg);
    QImage label = loadLabelAsGray(
        QDir(currentDatasetPath).filePath("labels/" + name), okLbl);

    if (!okImg || !okLbl) {
        logInvalidImage(name, "Falha de leitura");
        loadImageList();
        return;
    }
    selectCheck->setText(name);

    auto n_contrast = 2.0*double(contrast)/double(STEP_SIZE-1);
    auto n_brightness = int(511*double(brightness)/double(STEP_SIZE-1))-255;
    auto n_alphaThreshold = int(255*double(alphaThreshold)/double(STEP_SIZE-1));

    QImage rgba = rgb = adjustImage(rgb, n_brightness, n_contrast);
    rgba.setAlphaChannel(applyAlphaThreshold(label, n_alphaThreshold));

    imgRGB->setPixmap(QPixmap::fromImage(rgb));
    imgRGBA->setPixmap(QPixmap::fromImage(rgba));
    imgLabel->setPixmap(QPixmap::fromImage(label));

    drawPointsOverlay(imgRGB, name);
    drawPointsOverlay(imgRGBA, name);
    drawPointsOverlay(imgLabel, name);

    selectCheck->setChecked(selectedSet.contains(name));
    updateIndicators();

    gridViewer->clear();
    if (gridViewer->isVisible()) {
        QImage blank(DATASET_SIZE, DATASET_SIZE, QImage::Format_RGB888);
        blank.fill(Qt::white);
        auto center = gridViewer->parseName(name);
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                if (di == 0 && dj == 0) continue;
                int ti = center.i + di;
                int tj = center.j + dj;
                QString neighbor = center.prefix + "_" + QString::number(ti) +
                                   "_" + QString::number(tj) + ".tif";
                QImage nbr_img;
                bool img_exists = imageNames.contains(neighbor);
                bool is_invalid = invalidSet.contains(neighbor);
                bool is_loaded  = false;
                if (img_exists && !is_invalid) {
                    nbr_img = loadGDALImage(
                        QDir(currentDatasetPath).filePath("images/" + neighbor), is_loaded);
                }
                if (!is_loaded)
                    gridViewer->loadTile(blank, di, dj, DATASET_SIZE, .75);
                else
                    gridViewer->loadTile(nbr_img, di, dj, DATASET_SIZE, .75);
            }
        }
        gridViewer->loadTile(rgb, 0, 0, DATASET_SIZE);
        gridViewer->moveTo(0, 0, DATASET_SIZE);
    }

    refocusImageView();
}

void MainWindow::updateIndicators()
{
    indexEdit->setText(
        QString("%1/%2")
        .arg(currentVisiblePos+1)
        .arg(visibleIndices.size()));

    if (invalidSet.size() > 0)
        statsLabel->setText(
            QString("Imagens selecionadas: %1/%2 (%3 inválidas)")
            .arg(selectedSet.size())
            .arg(imageNames.size()+invalidSet.size())
            .arg(invalidSet.size()));
    else
        statsLabel->setText(
            QString("Imagens selecionadas: %1/%2")
            .arg(selectedSet.size())
            .arg(imageNames.size()));
}

void MainWindow::selectionToggled(bool checked)
{
    if (visibleIndices.isEmpty()) return;

    QString name = imageNames[visibleIndices[currentVisiblePos]];

    if (checked) selectedSet.insert(name);
    else selectedSet.remove(name);

    saveSelectedFile();
    if ((filterMode == FilterMode::Selected && !checked) ||
        (filterMode == FilterMode::Unselected && checked))
        rebuildVisibleIndices();
    else
        updateIndicators();
}

QImage MainWindow::loadGDALImage(const QString& path, bool& ok)
{
    ok = false;
    GDALDataset *ds =
        (GDALDataset*) GDALOpen(path.toStdString().c_str(), GA_ReadOnly);
    if (!ds || ds->GetRasterCount() < 3) return QImage();
    setBandCount(ds->GetRasterCount());

    //QImage img(DATASET_SIZE,DATASET_SIZE,QImage::Format_RGB888);
    //if (ds->RasterIO(GF_Read,0,0,DATASET_SIZE,DATASET_SIZE,
    //                 img.bits(),DATASET_SIZE,DATASET_SIZE,
    //                 GDT_Byte,3,nullptr,3,3*DATASET_SIZE,1) == CE_None)
    //    ok = true;

    // new code for load GDAL image
    std::vector<float> bandR, bandG, bandB;

    GDALRasterBand *bR = ds->GetRasterBand(spinR->value());
    GDALRasterBand *bG = ds->GetRasterBand(spinG->value());
    GDALRasterBand *bB = ds->GetRasterBand(spinB->value());

    bandR.resize(DATASET_SIZE*DATASET_SIZE);
    bandG.resize(DATASET_SIZE*DATASET_SIZE);
    bandB.resize(DATASET_SIZE*DATASET_SIZE);

    if (bR->RasterIO(GF_Read,0,0,DATASET_SIZE,DATASET_SIZE,bandR.data(),
                     DATASET_SIZE,DATASET_SIZE,GDT_Float32,0,0) == CE_None &&
        bG->RasterIO(GF_Read,0,0,DATASET_SIZE,DATASET_SIZE,bandG.data(),
                     DATASET_SIZE,DATASET_SIZE,GDT_Float32,0,0) == CE_None &&
        bB->RasterIO(GF_Read,0,0,DATASET_SIZE,DATASET_SIZE,bandB.data(),
                     DATASET_SIZE,DATASET_SIZE,GDT_Float32,0,0) == CE_None)
        ok = true;

    bool norm = normSelector->isChecked();

    std::vector<uint8_t> R8, G8, B8;

    if (norm) {
        normalizeToByte(bandR, R8);
        normalizeToByte(bandG, G8);
        normalizeToByte(bandB, B8);
    } else {
        // cast direto
        R8.assign(bandR.begin(), bandR.end());
        G8.assign(bandG.begin(), bandG.end());
        B8.assign(bandB.begin(), bandB.end());
    }

    QImage img(DATASET_SIZE, DATASET_SIZE, QImage::Format_RGB888);
    for (int i=0;i<DATASET_SIZE*DATASET_SIZE;i++) {
        img.bits()[3*i+0] = R8[i];
        img.bits()[3*i+1] = G8[i];
        img.bits()[3*i+2] = B8[i];
    }
    // end of new code

    GDALClose(ds);
    return img;
}

QImage MainWindow::loadLabelAsGray(const QString& path, bool& ok)
{
    ok = false;
    GDALDataset *ds =
        (GDALDataset*) GDALOpen(path.toStdString().c_str(), GA_ReadOnly);
    if (!ds || ds->GetRasterCount() < 1) return QImage();

    QImage img(DATASET_SIZE,DATASET_SIZE,QImage::Format_Grayscale8);
    if (ds->RasterIO(GF_Read,0,0,DATASET_SIZE,DATASET_SIZE,
                     img.bits(),DATASET_SIZE,DATASET_SIZE,
                     GDT_Byte,1,nullptr,1,DATASET_SIZE,1) == CE_None)
        ok = true;

    GDALClose(ds);
    return img;
}

QImage MainWindow::applyAlphaThreshold(const QImage& label, int& threshold)
{
    QImage a = label.copy();
    for (int y=0;y<a.height();++y) {
        uchar *l = a.scanLine(y);
        for (int x=0;x<a.width();++x)
            l[x] = (l[x]>=threshold)?l[x]:threshold;
    }
    return a;
}

QImage MainWindow::adjustImage(const QImage& img, int& brightness, double& contrast)
{
    // brightness: -255 <-> 255
    // contrast: 0.0 <-> 2.0

    QImage newImage = img.convertToFormat(QImage::Format_RGBA8888);
    int pixels = newImage.width() * newImage.height();
    unsigned int *data = (unsigned int *)newImage.bits();

    for (int i = 0; i < pixels; ++i) {
        unsigned int p = data[i];

        int r = qRed(p);
        int g = qGreen(p);
        int b = qBlue(p);

        r = qBound(0, (int)((r - 128) * contrast + 128) + brightness, 255);
        g = qBound(0, (int)((g - 128) * contrast + 128) + brightness, 255);
        b = qBound(0, (int)((b - 128) * contrast + 128) + brightness, 255);

        data[i] = qRgba(r, g, b, 0);
    }
    return newImage.convertToFormat(QImage::Format_RGB888);
}

void MainWindow::refocusImageView()
{
    imageView->setFocus(Qt::OtherFocusReason);
}

void MainWindow::addPoint(const QString& imageName, const QPointF& p)
{
    imagePoints[imageName].append(p);
    savePoints();
}

void MainWindow::loadPoints()
{
    imagePoints.clear();

    QFile f(QDir(currentDatasetPath).filePath("points.txt"));
    if (!f.open(QIODevice::ReadOnly)) return;

    QTextStream in(&f);
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;

        auto parts = line.split(';');
        if (parts.size() < 3) continue;

        QString img = parts[0];
        bool ok1, ok2;
        float x = parts[1].toFloat(&ok1);
        float y = parts[2].toFloat(&ok2);

        if (ok1 && ok2)
            imagePoints[img].append(QPointF(x,y));
    }
}

void MainWindow::removeNearestPoint(const QString& imageName,
                                    const QPointF& p,
                                    float radiusPx = 20)
{
    if (!imagePoints.contains(imageName)) return;

    auto &vec = imagePoints[imageName];
    int best = -1;
    float bestDist = radiusPx;

    for (int i = 0; i < vec.size(); ++i) {
        float d = QLineF(vec[i], p).length();
        if (d < bestDist) {
            bestDist = d;
            best = i;
        }
    }

    if (best >= 0) {
        vec.remove(best);
        savePoints();
    }
}

void MainWindow::savePoints()
{
    QFile f(QDir(currentDatasetPath).filePath("points.txt"));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;

    QTextStream out(&f);
    for (auto it = imagePoints.begin(); it != imagePoints.end(); ++it) {
        const QString& img = it.key();
        for (const QPointF& p : it.value()) {
            out << img << ";" << p.x() << ";" << p.y() << "\n";
        }
    }
}

void MainWindow::drawPointsOverlay(QLabel *label, const QString& imgName)
{
    if (!imagePoints.contains(imgName)) return;

    QPixmap pix = label->pixmap();
    QPainter p(&pix);

    p.setPen(QPen(Qt::red, 2));
    for (const QPointF& pt : imagePoints[imgName]) {
        p.drawEllipse(QPointF(pt.x(), pt.y()), 3, 3);
    }

    label->setPixmap(pix);
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    if ((obj == imgRGB || obj == imgRGBA || obj == imgLabel) &&
            event->type() == QEvent::MouseButtonPress &&
            !visibleIndices.isEmpty()) {

        QMouseEvent *e = static_cast<QMouseEvent*>(event);

        QPoint pos = e->position().toPoint();

        QString imgName = imageNames[visibleIndices[currentVisiblePos]];

        if (e->button() == Qt::LeftButton) {
            addPoint(imgName, QPointF(pos.x(), pos.y()));
        }
        else if (e->button() == Qt::RightButton) {
            removeNearestPoint(imgName, QPointF(pos.x(), pos.y()));
        }
        loadCurrentImage();

        return true;
    }

    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::setBandCount(int n)
{
    for (auto *s : {spinR, spinG, spinB}) {
        s->setMaximum(n);
    }
}

bool MainWindow::needNormalization(const std::vector<float>& v)
{
    float minv, maxv;
    minv = std::numeric_limits<float>::max();
    maxv = std::numeric_limits<float>::lowest();
    for (float x : v) {
        if (!std::isfinite(x)) continue;
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);
    }
    return (minv < 0.0f || maxv > 255.0f);
}

void MainWindow::normalizeToByte(const std::vector<float>& src,
                                 std::vector<uint8_t>& dst)
{
    float minv, maxv;
    minv = std::numeric_limits<float>::max();
    maxv = std::numeric_limits<float>::lowest();
    for (float x : src) {
        if (!std::isfinite(x)) continue;
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);
    }
    dst.resize(src.size());
    float range = maxv - minv;
    if (range <= 0.0f) range = 1.0f;

    for (size_t i=0;i<src.size();++i) {
        float v = (src[i] - minv) / range;
        v = std::clamp(v, 0.0f, 1.0f);
        dst[i] = static_cast<uint8_t>(v * 255.0f);
    }
}

void MainWindow::loadViewerPresets()
{
    viewerPresets.clear();

    QFile f(rootPath + "/presets.txt");
    if (!f.open(QIODevice::ReadOnly)) {
        qWarning() << "presets.txt não encontrado em" << rootPath;
        return;
    }

    QTextStream in(&f);
    int lineCount = 0;

    while (!in.atEnd() && lineCount < 9) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;
        if (line.startsWith("#")) continue;

        QStringList parts = line.split(QRegularExpression("\\s+"));
        if (parts.size() < 4) continue;

        bool ok1, ok2, ok3, ok4;

        int r = parts[0].toInt(&ok1);
        int g = parts[1].toInt(&ok2);
        int b = parts[2].toInt(&ok3);

        QString normStr = parts[3].toLower();
        bool norm;

        if (normStr == "true" || normStr == "1") {
            norm = true; ok4 = true;
        } else if (normStr == "false" || normStr == "0") {
            norm = false; ok4 = true;
        } else {
            ok4 = false;
        }

        if (!(ok1 && ok2 && ok3 && ok4)) continue;

        viewerPresets.push_back({r,g,b,norm});
        lineCount++;
    }

    /* ====== Shortcuts 1..9 ====== */
    for (int i = 0; i < viewerPresets.size(); ++i) {

        QShortcut *sc = new QShortcut(QKeySequence(QString::number(i+1)), this);

        connect(sc, &QShortcut::activated, this, [this, i]() {
            if (i >= viewerPresets.size()) return;

            const ViewerPreset &p = viewerPresets[i];

            spinR->blockSignals(true);
            spinR->setValue(p.r);
            spinG->blockSignals(true);
            spinG->setValue(p.g);
            spinB->blockSignals(true);
            spinB->setValue(p.b);
            normSelector->blockSignals(true);
            normSelector->setChecked(p.norm);

            spinR->blockSignals(false);
            spinG->blockSignals(false);
            spinB->blockSignals(false);
            normSelector->blockSignals(false);

            loadCurrentImage();
        });
    }
}

void MainWindow::resizeToFitImageView()
{
    if (!imageView || !imageView->layout())
        return;

    QSize contentSize = imageView->layout()->sizeHint();
    QMargins m = imageView->layout()->contentsMargins();
    contentSize.rwidth()  += m.left() + m.right();
    contentSize.rheight() += m.top()  + m.bottom();
    int top    = menuBar()   ? menuBar()->height()   : 0;
    int bottom = statusBar() ? statusBar()->height() : 0;
    int left = 0, right = 0;
    for (QToolBar* tb : findChildren<QToolBar*>()) {
        if (tb->isFloating() || !tb->isVisible()) continue;
        Qt::ToolBarArea area = toolBarArea(tb);
        switch (area) {
            case Qt::TopToolBarArea:    top += tb->height();    break;
            case Qt::BottomToolBarArea: bottom += tb->height(); break;
            case Qt::LeftToolBarArea:   left += tb->width();    break;
            case Qt::RightToolBarArea:  right += tb->width();   break;
            default: break;
        }
    }
    for (auto dock : findChildren<QDockWidget*>()) {
        if (dock->isFloating() || dock->isHidden()) continue;
        switch (dockWidgetArea(dock)) {
            case Qt::LeftDockWidgetArea:   left  += dock->width();  break;
            case Qt::RightDockWidgetArea:  right += dock->width();  break;
            case Qt::TopDockWidgetArea:    top   += dock->height(); break;
            case Qt::BottomDockWidgetArea: bottom+= dock->height(); break;
            default: break;
        }
    }
    QSize desired = contentSize + QSize(left+right, top+bottom);

    QRect screenGeom = screen()->availableGeometry();
    int maxW = screenGeom.width()  * 0.95;
    int maxH = screenGeom.height() * 0.95;

    QSize finalSize = desired;
    if (finalSize.width() > maxW)
        finalSize.setWidth(maxW);
    if (finalSize.height() > maxH)
        finalSize.setHeight(maxH);
    resize(finalSize);
}
