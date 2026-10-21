#include "gridviewerwidget.h"

#include <QDir>
#include <QGraphicsPixmapItem>
#include <QRegularExpression>
#include <QImageReader>

#define GRID_GAP 3

GridViewer::GridViewer(QWidget *parent)
    : QGraphicsView(parent)
{
    scene = new QGraphicsScene(this);
    setScene(scene);

    setRenderHint(QPainter::SmoothPixmapTransform);
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(AnchorViewCenter);
    setResizeAnchor(AnchorViewCenter);
    setAlignment(Qt::AlignCenter);
}

void GridViewer::clear()
{
    scene->clear();
}

TileInfo GridViewer::parseName(const QString& name) const
{
    QRegularExpression rx("^(.*)_([0-9]+)_([0-9]+)\\.tif$");
    auto m = rx.match(name);

    TileInfo info;
    if (!m.hasMatch()) return info;

    info.prefix = m.captured(1);
    info.i = m.captured(2).toInt();
    info.j = m.captured(3).toInt();
    return info;
}

void GridViewer::loadTile(const QImage& img, int i, int j, int tileSize, float opacity)
{
    QPixmap px = QPixmap::fromImage(img);

    auto *item = new QGraphicsPixmapItem(px.scaled(
        tileSize, tileSize,
        Qt::IgnoreAspectRatio,
        Qt::SmoothTransformation
    ));

    int x = (j + 1) * (tileSize + GRID_GAP);
    int y = (i + 1) * (tileSize + GRID_GAP);

    item->setPos(x, y);
    item->setOpacity(opacity);
    scene->addItem(item);
}

void GridViewer::moveTo(int i, int j, int tileSize)
{
    int x = (j + 1) * (tileSize + GRID_GAP) + (tileSize + GRID_GAP)/2;
    int y = (i + 1) * (tileSize + GRID_GAP) + (tileSize + GRID_GAP)/2;
    centerOn(x,y);
}
