#pragma once

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QMap>
#include <QSet>

struct TileInfo {
    QString prefix;
    int i;
    int j;
};

class GridViewer : public QGraphicsView
{
    Q_OBJECT

public:
    explicit GridViewer(QWidget *parent = nullptr);

    void clear();
    void loadTile(const QImage& img, int i, int j, int tileSize, float opacity = 1.0);
    void moveTo(int i, int j, int tileSize);
    TileInfo parseName(const QString& name) const;

private:
    QGraphicsScene *scene;
};
