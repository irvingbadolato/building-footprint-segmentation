#pragma once

#include <QMainWindow>
#include <QComboBox>
#include <QLabel>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QAction>
#include <QEvent>
#include <QSet>
#include <QMap>
#include <QPointF>
#include <QVector>
#include "gridviewerwidget.h"

enum class FilterMode {
    All = 0,
    Selected,
    Unselected
};

struct ViewerPreset {
    int r;
    int g;
    int b;
    bool norm;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(const QString& rootPath, QWidget *parent = nullptr);

signals:
    void bandsChanged(int r, int g, int b);

private slots:
    void onDatasetChanged(int index);
    void upMixture();
    void downMixture();
    void upBrightness();
    void downBrightness();
    void upContrast();
    void downContrast();
    void nextImage();
    void prevImage();
    void selectionToggled(bool checked);

private:
    // ---------- UI ----------
    GridViewer  *gridViewer;
    QComboBox   *datasetCombo;
    QComboBox   *filterCombo;
    QDockWidget *dockArea;
    QLabel      *imgRGB;
    QLabel      *imgRGBA;
    QLabel      *imgLabel;
    QLabel      *statsLabel;
    QLayout     *imageLayout;
    QCheckBox   *selectCheck;
    QCheckBox   *normSelector;
    QSlider     *alphaSlider;
    QSlider     *brightnessSlider;
    QSlider     *contrastSlider;
    QLineEdit   *indexEdit;
    QPushButton *prevBtn;
    QPushButton *nextBtn;
    QSpinBox    *spinR;
    QSpinBox    *spinG;
    QSpinBox    *spinB;
    QScrollArea *mainViewer;
    QWidget     *imageView;
    bool         isVertical;

    // ---------- Actions ----------
    QAction *atcUpMix;
    QAction *actDnMix;
    QAction *actUpBri;
    QAction *actDnBri;
    QAction *actUpCon;
    QAction *actDnCon;
    QAction *actNext;
    QAction *actPrev;
    QAction *actToggleSelect;
    QAction *actChangeImageViewer;
    QAction *actExchangeViewers;
    QAction *actShowGrid;

    // ---------- Paths ----------
    QString rootPath;
    QString currentDatasetPath;

    // ---------- Image state ----------
    QStringList imageNames;          // válidas
    QVector<int> visibleIndices;     // filtradas
    int currentVisiblePos = 0;

    // ---------- Persistent state ----------
    QSet<QString> selectedSet;
    QSet<QString> invalidSet;
    QMap<QString, QVector<QPointF>> imagePoints;
    QVector<ViewerPreset> viewerPresets;

    // ---------- Modes ----------
    FilterMode filterMode = FilterMode::All;

    // ---------- DIP values ----------
    int alphaThreshold;
    int brightness;
    int contrast;

    // ---------- Core ----------
    void createActions();
    void scanDatasets();
    void loadImageList();
    void rebuildVisibleIndices();
    void loadCurrentImage();
    void jumpToIndex(int idx);
    void updateIndicators();
    void refocusImageView();

    // ---------- Persistence ----------
    void loadSelectedFile();
    void saveSelectedFile();
    void loadInvalidFile();
    void logInvalidImage(const QString& name, const QString& reason);
    void loadPoints();
    void savePoints();
    void loadViewerPresets();

    // ---------- Helpers ----------
    bool eventFilter(QObject *obj, QEvent *event) override;
    void resizeToFitImageView();
    bool isSelected(const QString& name) const;
    QImage loadGDALImage(const QString& path, bool& ok);
    QImage loadLabelAsGray(const QString& path, bool& ok);
    QImage applyAlphaThreshold(const QImage& label, int& threshold);
    QImage adjustImage(const QImage& img, int& brightness, double& contrast);
    void addPoint(const QString& imageName, const QPointF& p);
    void removeNearestPoint(const QString& imageName, const QPointF& p, float radiusPx);
    void drawPointsOverlay(QLabel *label, const QString& imgName);
    void setVertical(bool orientation);
    bool needNormalization(const std::vector<float>& v);
    void normalizeToByte(const std::vector<float>& src,std::vector<uint8_t>& dst);
    void setBandCount(int n);
};

